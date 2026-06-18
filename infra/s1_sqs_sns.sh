#!/usr/bin/env bash
# =============================================================================
# s1_sqs_sns.sh -- S1 ingest backbone, Infrastructure-as-Code (idempotent).
#
# Creates (or reconciles) the AWS resources the S1 satellite-ingest worker
# long-polls: an SQS main queue + a dead-letter queue, subscribed to NOAA's
# public Open Data SNS topic NewGOES19Object so a new GOES-19 object fires an
# event the worker renders into the R2 /shadow/ prefix.
#
# Run once from anywhere with the `tat-sat-ingest` IAM creds in the env
# (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION=us-east-1).
# Re-running is safe: every step is create-or-update.
#
#   ./infra/s1_sqs_sns.sh
#
# -----------------------------------------------------------------------------
# THE LOAD-BEARING CORRECTNESS DETAIL (SATELLITE-REVIEW INGEST-1):
#   NOAA's NewGOES19Object delivers an S3-event payload whose object key lives
#   in the message BODY (Records[].s3.object.key) -- there are NO usable message
#   attributes. SNS subscription filter policies DEFAULT to
#   FilterPolicyScope=MessageAttributes, which on an attribute-less message
#   matches NOTHING and SILENTLY DROPS EVERY EVENT. We MUST set
#   FilterPolicyScope=MessageBody with a body-path prefix policy. The §3.5
#   acceptance check (infra/s1_acceptance_check.sh) proves the filter is not a
#   silent no-op: messages actually arrive AND ~nothing wanted is filtered out.
# =============================================================================
set -euo pipefail

REGION="${AWS_DEFAULT_REGION:-us-east-1}"
TOPIC_ARN="${S1_TOPIC_ARN:-arn:aws:sns:us-east-1:123901341784:NewGOES19Object}"
QUEUE_NAME="${S1_QUEUE_NAME:-tat-sat-goes19-cmip}"
DLQ_NAME="${S1_DLQ_NAME:-tat-sat-goes19-cmip-dlq}"
MAX_RECEIVE="${S1_MAX_RECEIVE_COUNT:-5}"
# Visibility timeout must exceed worst-case fetch+render so a slow render is not
# redelivered mid-flight (SATELLITE-REARCH §3.1/§11-F). S1's clean-IR meso
# render is fast (~2-5 s), but size generously: a true-color render is tens of
# seconds, and idempotent keys (§3.3) make an occasional redelivery merely
# wasteful, never wrong.
VISIBILITY_TIMEOUT="${S1_VISIBILITY_TIMEOUT:-120}"
# 14 days = the SQS max. A sustained R2/box outage keeps the backlog instead of
# silently aging it out before the box recovers (the §3.4 + §3.6 self-heal
# leans on the backlog still being there).
RETENTION="${S1_RETENTION:-1209600}"
LONGPOLL_WAIT="${S1_LONGPOLL_WAIT:-20}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILTER_POLICY_FILE="${HERE}/s1_filter_policy.json"

echo ">> region=${REGION}  topic=${TOPIC_ARN}"
echo ">> main_queue=${QUEUE_NAME}  dlq=${DLQ_NAME}  maxReceiveCount=${MAX_RECEIVE}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
echo ">> account=${ACCOUNT_ID}"

mktmp() { mktemp "${TMPDIR:-/tmp}/s1iac.XXXXXX"; }

# --- 1. DLQ ------------------------------------------------------------------
echo ">> [1/5] dead-letter queue ${DLQ_NAME}"
DLQ_URL="$(aws sqs create-queue --queue-name "${DLQ_NAME}" \
  --attributes "MessageRetentionPeriod=${RETENTION}" \
  --query QueueUrl --output text)"
DLQ_ARN="$(aws sqs get-queue-attributes --queue-url "${DLQ_URL}" \
  --attribute-names QueueArn --query 'Attributes.QueueArn' --output text)"
echo "   dlq_url=${DLQ_URL}"
echo "   dlq_arn=${DLQ_ARN}"

# --- 2. Main queue -----------------------------------------------------------
echo ">> [2/5] main queue ${QUEUE_NAME}"
QUEUE_URL="$(aws sqs create-queue --queue-name "${QUEUE_NAME}" \
  --query QueueUrl --output text)"
QUEUE_ARN="$(aws sqs get-queue-attributes --queue-url "${QUEUE_URL}" \
  --attribute-names QueueArn --query 'Attributes.QueueArn' --output text)"
echo "   queue_url=${QUEUE_URL}"
echo "   queue_arn=${QUEUE_ARN}"

# --- 3. Main-queue attributes: visibility/retention/longpoll + redrive->DLQ --
echo ">> [3/5] main-queue attributes (+ redrive to DLQ, maxReceiveCount=${MAX_RECEIVE})"
ATTRS_FILE="$(mktmp)"
python3 - "${DLQ_ARN}" "${MAX_RECEIVE}" "${VISIBILITY_TIMEOUT}" "${RETENTION}" "${LONGPOLL_WAIT}" \
  > "${ATTRS_FILE}" <<'PY'
import json, sys
dlq_arn, maxrecv, vis, ret, wait = sys.argv[1:6]
print(json.dumps({
    "VisibilityTimeout": vis,
    "MessageRetentionPeriod": ret,
    "ReceiveMessageWaitTimeSeconds": wait,
    "RedrivePolicy": json.dumps({
        "deadLetterTargetArn": dlq_arn,
        "maxReceiveCount": int(maxrecv),
    }),
}))
PY
aws sqs set-queue-attributes --queue-url "${QUEUE_URL}" --attributes "file://${ATTRS_FILE}"
rm -f "${ATTRS_FILE}"

# --- 4. Main-queue access policy: allow ONLY this SNS topic to SendMessage ----
# Cross-account authz we owe (SATELLITE-REARCH §5.7): NOAA's topic is public, so
# the authz is a policy on OUR queue -- Principal sns.amazonaws.com, action
# sqs:SendMessage, aws:SourceArn pinned to the topic ARN (no broader principal).
echo ">> [4/5] main-queue access policy (SNS principal, SourceArn pinned to topic)"
POLICY_FILE="$(mktmp)"
python3 - "${QUEUE_ARN}" "${TOPIC_ARN}" > "${POLICY_FILE}" <<'PY'
import json, sys
qarn, topic = sys.argv[1], sys.argv[2]
policy = {"Version": "2012-10-17", "Statement": [{
    "Sid": "AllowNoaaSnsTopicSend",
    "Effect": "Allow",
    "Principal": {"Service": "sns.amazonaws.com"},
    "Action": "sqs:SendMessage",
    "Resource": qarn,
    "Condition": {"ArnEquals": {"aws:SourceArn": topic}},
}]}
print(json.dumps({"Policy": json.dumps(policy)}))
PY
aws sqs set-queue-attributes --queue-url "${QUEUE_URL}" --attributes "file://${POLICY_FILE}"
rm -f "${POLICY_FILE}"

# --- 5. SNS subscription: MessageBody filter scope + body-path prefix policy --
echo ">> [5/5] subscribe queue to ${TOPIC_ARN}"
SUB_ARN="$(aws sns subscribe --topic-arn "${TOPIC_ARN}" --protocol sqs \
  --notification-endpoint "${QUEUE_ARN}" --return-subscription-arn --output text)"
echo "   subscription_arn=${SUB_ARN}"

# Build subscription attributes from the readable filter-policy file. EVERY value
# must be a STRING (FilterPolicy is a stringified JSON). RawMessageDelivery=true
# so SQS receives the bare S3 event the worker parses (no SNS envelope to unwrap).
SUB_ATTRS_FILE="$(mktmp)"
python3 - "${FILTER_POLICY_FILE}" > "${SUB_ATTRS_FILE}" <<'PY'
import json, sys
policy = json.load(open(sys.argv[1]))
# ORDER MATTERS: FilterPolicyScope MUST be set to MessageBody BEFORE FilterPolicy.
# AWS rejects a nested body policy while the scope is still the default
# MessageAttributes ("Filter policy scope MessageAttributes does not support
# nested filter policy") -- the INGEST-1 trap, surfaced by the API itself.
print(json.dumps({
    "FilterPolicyScope": "MessageBody",      # <-- FIRST: NOT the default MessageAttributes
    "FilterPolicy": json.dumps(policy, separators=(",", ":")),
    "RawMessageDelivery": "true",
}))
PY
# set-subscription-attributes takes ONE attribute at a time; apply all three.
python3 - "${SUB_ATTRS_FILE}" "${SUB_ARN}" <<'PY'
import json, subprocess, sys
attrs = json.load(open(sys.argv[1]))
sub = sys.argv[2]
for name, value in attrs.items():
    subprocess.run([
        "aws", "sns", "set-subscription-attributes",
        "--subscription-arn", sub,
        "--attribute-name", name,
        "--attribute-value", value,
    ], check=True)
    print(f"   set {name}")
PY
rm -f "${SUB_ATTRS_FILE}"

cat <<EOF

=============================================================================
S1 ingest infra ready.
  TOPIC_ARN        = ${TOPIC_ARN}
  QUEUE_URL        = ${QUEUE_URL}
  QUEUE_ARN        = ${QUEUE_ARN}
  DLQ_URL          = ${DLQ_URL}
  DLQ_ARN          = ${DLQ_ARN}
  SUBSCRIPTION_ARN = ${SUB_ARN}
  FilterPolicyScope= MessageBody   (body-path Records[].s3.object.key prefix ABI-L2-CMIPM/)
  maxReceiveCount  = ${MAX_RECEIVE}  -> DLQ

Next: prove the filter is not a silent no-op:
  SUB_ARN='${SUB_ARN}' QUEUE_URL='${QUEUE_URL}' ./infra/s1_acceptance_check.sh

Worker .env needs:  S1_QUEUE_URL=${QUEUE_URL}
=============================================================================
EOF
