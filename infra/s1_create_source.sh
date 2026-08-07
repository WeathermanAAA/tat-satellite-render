#!/usr/bin/env bash
# =============================================================================
# s1_create_source.sh <goes19|goes18|himawari9> -- create/reconcile the SQS
# queue + DLQ + NOAA SNS subscription for ONE S1 source, pulling its topic /
# queue / DLQ / body-path filter prefix straight from s1_sources.py (the single
# source of truth) and handing them to the idempotent s1_sqs_sns.sh.
#
# Run once per source from anywhere with the tat-sat-ingest IAM creds in the env
# (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION=us-east-1).
# Re-running is safe (every AWS step is create-or-update).
#
#   ./infra/s1_create_source.sh goes18
#   ./infra/s1_create_source.sh himawari9
# =============================================================================
set -euo pipefail
SRC="${1:?usage: s1_create_source.sh <goes19|goes18|himawari9>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
cd "${REPO}"

eval "$(python3 - "$SRC" <<'PY'
import sys
import s1_sources as SRC
s = SRC.get_source(sys.argv[1])
for k, v in (("S1_TOPIC_ARN", s.topic_arn), ("S1_QUEUE_NAME", s.queue_name),
             ("S1_DLQ_NAME", s.dlq_name), ("S1_FILTER_PREFIX", s.filter_prefix)):
    print(f"export {k}={v}")
PY
)"

echo ">> S1 infra for source=${SRC}"
echo ">>   topic=${S1_TOPIC_ARN}"
echo ">>   queue=${S1_QUEUE_NAME}  dlq=${S1_DLQ_NAME}  filter=${S1_FILTER_PREFIX}"
exec "${HERE}/s1_sqs_sns.sh"
