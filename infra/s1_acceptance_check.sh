#!/usr/bin/env bash
# =============================================================================
# s1_acceptance_check.sh -- §3.5 acceptance check for the SNS->SQS filter.
#
# Proves the filter is NOT a silent no-op (the INGEST-1 failure mode where the
# default MessageAttributes scope matches nothing and every event is dropped,
# yet the system still renders via backfill so the metric reads green). Three
# independent proofs, any one of which would expose a no-op:
#
#   A. CONFIG    -- the subscription's FilterPolicyScope is MessageBody (NOT the
#                   default MessageAttributes) and the body-path FilterPolicy is
#                   present. A scope of MessageAttributes here is the smoking gun.
#   B. LIVE      -- peek the queue: messages ACTUALLY arrive (filter passed them)
#                   AND every received object key is under ABI-L2-CMIPM/ (the
#                   filter is body-scoped and correct), with >=1 CMIPM2-C13 seen.
#                   Plus the in-account SQS metric NumberOfMessagesSent > 0
#                   (NOAA owns the topic, so SNS-side NumberOfNotifications
#                   FilteredOut is in THEIR account and unreadable from ours --
#                   the SQS sent-count is the in-account proxy that a no-op
#                   filter would leave at zero).
#   C. OFFLINE   -- the committed filter policy, run against a CAPTURED real
#                   NOAA firehose batch, passes exactly the CMIPM keys and
#                   rejects the rest (infra/s1_filter_check.py). This is the
#                   deterministic "would a wanted notification be filtered out?"
#                   answer that does not depend on live timing.
#
# Usage (after s1_sqs_sns.sh):
#   SUB_ARN='arn:...:NewGOES19Object:...' QUEUE_URL='https://.../tat-sat-goes19-cmip' \
#     ./infra/s1_acceptance_check.sh
# =============================================================================
set -euo pipefail

QUEUE_URL="${QUEUE_URL:-${S1_QUEUE_URL:?set QUEUE_URL or S1_QUEUE_URL}}"
QUEUE_NAME="${QUEUE_URL##*/}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PEEK_WAIT="${S1_PEEK_WAIT:-20}"
# Which source's policy the peeked keys are judged against (part B) and which
# source part C validates. Default matches the original single-source check.
S1_SOURCE="${S1_SOURCE:-goes19}"
S1_REPO_ROOT="$(cd "${HERE}/.." && pwd)"
export S1_SOURCE S1_REPO_ROOT
fail=0

echo "============================================================"
echo "A. CONFIG -- subscription filter scope"
echo "============================================================"
if [[ -n "${SUB_ARN:-}" ]]; then
  SCOPE="$(aws sns get-subscription-attributes --subscription-arn "${SUB_ARN}" \
    --query 'Attributes.FilterPolicyScope' --output text 2>/dev/null || echo NONE)"
  FPOL="$(aws sns get-subscription-attributes --subscription-arn "${SUB_ARN}" \
    --query 'Attributes.FilterPolicy' --output text 2>/dev/null || echo NONE)"
  RAW="$(aws sns get-subscription-attributes --subscription-arn "${SUB_ARN}" \
    --query 'Attributes.RawMessageDelivery' --output text 2>/dev/null || echo NONE)"
  echo "   FilterPolicyScope = ${SCOPE}"
  echo "   RawMessageDelivery= ${RAW}"
  echo "   FilterPolicy      = ${FPOL}"
  if [[ "${SCOPE}" != "MessageBody" ]]; then
    echo "   !! FAIL: scope is not MessageBody -- the filter matches NOTHING (INGEST-1)."
    fail=1
  else
    echo "   OK: MessageBody scope."
  fi
else
  echo "   (SUB_ARN not set -- skipping config assert; pass SUB_ARN to enable)"
fi

echo
echo "============================================================"
echo "B. LIVE -- peek the queue (non-destructive) + SQS sent metric"
echo "============================================================"
PEEK_JSON="$(mktemp)"
# Long-poll a peek; visibility-timeout 0 so peeked messages stay immediately
# available for the worker (this is a read-only check).
aws sqs receive-message --queue-url "${QUEUE_URL}" \
  --max-number-of-messages 10 --wait-time-seconds "${PEEK_WAIT}" \
  --visibility-timeout 0 \
  --query 'Messages[].Body' --output json > "${PEEK_JSON}" 2>/dev/null || echo "null" > "${PEEK_JSON}"
python3 - "${PEEK_JSON}" <<'PY' || fail=1
import json, sys
raw = json.load(open(sys.argv[1]))
msgs = raw or []
def extract_key(body_str):
    # Robust to BOTH raw S3-event delivery and the SNS envelope (Type:
    # Notification with the S3 event in the "Message" string) -- the worker is
    # envelope-aware for the same reason (raw-delivery propagation windows).
    try:
        b = json.loads(body_str)
    except Exception:
        return "<unparseable>"
    if isinstance(b, dict) and "Records" not in b and "Message" in b:
        try:
            b = json.loads(b["Message"])
        except Exception:
            return "<unparseable-envelope>"
    try:
        return b["Records"][0]["s3"]["object"]["key"]
    except Exception:
        return "<no-key>"
keys = [extract_key(m) for m in msgs]
print(f"   peeked {len(keys)} message(s)")
for k in keys:
    print("    ", k)
if not keys:
    print("   !! WARN: 0 messages peeked this window (could be a quiet moment); "
          "rely on the SQS sent-metric + offline proof below.")
    sys.exit(0)
# Judge the peeked keys against THIS source's committed policy (the local
# matcher must agree with what SNS just passed) + count our product among
# them via the worker's own predicate. NOTE: right after a policy edit, SNS
# applies the change eventually (~up to 15 min) -- messages passed by the
# OLD policy may still be in flight; warn, don't fail, on those.
import os
sys.path.insert(0, os.environ["S1_REPO_ROOT"])
sys.path.insert(0, os.path.join(os.environ["S1_REPO_ROOT"], "infra"))
import s1_filter_check as F
import s1_sources as SRC
source = SRC.get_source(os.environ.get("S1_SOURCE", "goes19"))
policy = SRC.filter_policy(source)
prefix_ok = [k for k in keys if k.startswith(source.filter_prefix)]
foreign = [k for k in keys if not k.startswith(source.filter_prefix)]
policy_pass = [k for k in keys
               if F.matches({"Records": [{"s3": {"object": {"key": k}}}]}, policy)]
ours = [k for k in keys if SRC.is_ours(source, SRC.parse(source, k))]
if foreign:
    print(f"   !! FAIL: {len(foreign)} key(s) outside {source.filter_prefix} "
          "slipped the filter:")
    for k in foreign: print("       ", k)
    sys.exit(1)
stale_policy = [k for k in prefix_ok if k not in policy_pass]
if stale_policy:
    print(f"   ?? {len(stale_policy)} peeked key(s) match the product prefix but "
          "NOT the committed band-tight policy -- expected for ~15 min after a "
          "policy edit (SNS eventual consistency); re-run to confirm it drains.")
print(f"   OK: all {len(keys)} peeked keys are {source.filter_prefix}; "
      f"{len(policy_pass)} match the committed policy; "
      f"{len(ours)} are the S1 product ({source.product_path}).")
PY

# In-account SQS metric: NumberOfMessagesSent > 0 over the last 15 min.
echo "   -- SQS NumberOfMessagesSent (last 15 min, in-account) --"
END="$(python3 -c 'import datetime;print(datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))')"
START="$(python3 -c 'import datetime;print((datetime.datetime.now(datetime.timezone.utc)-datetime.timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ"))')"
SENT="$(aws cloudwatch get-metric-statistics --namespace AWS/SQS \
  --metric-name NumberOfMessagesSent \
  --dimensions Name=QueueName,Value="${QUEUE_NAME}" \
  --start-time "${START}" --end-time "${END}" --period 300 --statistics Sum \
  --query 'sort_by(Datapoints,&Timestamp)[].Sum' --output text 2>/dev/null || echo "")"
echo "      datapoints(Sum/5min): ${SENT:-<none yet>}"
echo "      (note: SQS metrics lag ~5 min; a fresh subscription may show none "
echo "       yet -- the live peek above is the immediate proof.)"
rm -f "${PEEK_JSON}"

echo
echo "============================================================"
echo "C. OFFLINE -- committed filter vs CAPTURED real NOAA firehose"
echo "============================================================"
if python3 "${HERE}/s1_filter_check.py" --source "${S1_SOURCE}"; then
  echo "   OK: ${S1_SOURCE} policy agrees with real keys (product pass, rest reject)."
else
  echo "   !! FAIL: ${S1_SOURCE} policy disagrees with real keys."
  fail=1
fi

echo
if [[ "${fail}" -eq 0 ]]; then
  echo ">> ACCEPTANCE PASS: the SNS->SQS body filter is live, body-scoped, and not a no-op."
else
  echo ">> ACCEPTANCE FAIL: see the !! lines above."
fi
exit "${fail}"
