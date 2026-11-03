#!/usr/bin/env bash
# Write results/<run>/RUN.md: a machine-generated run card (what ran, on what, and how each step went).
# Regenerated after every step so even an interrupted run has one. Narrative notes go in LOGBOOK.md.
#   bench/runcard.sh results/<run>
RUN=$1
cd "$(dirname "$0")/.." || exit 1
TOKEN=$(curl -s -m 2 -X PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
md() { curl -s -m 2 -H "X-aws-ec2-metadata-token: $TOKEN" "http://169.254.169.254/latest/meta-data/$1"; }
ver() { grep -m1 -E "^$1 " "$RUN/inventory.log" 2>/dev/null | awk '{print $2}'; }

{
  echo "# Run $(basename "$RUN")"
  echo
  echo "| | |"
  echo "|---|---|"
  echo "| Instance | $(md instance-id) ($(md instance-type), $(md placement/availability-zone)) |"
  echo "| AMI | $(md ami-id) |"
  echo "| Repo commit | $(git log -1 --format='%h %s' -- bench infra 2>/dev/null) |"
  echo "| Venvs | $(cat "$RUN/venvs.txt" 2>/dev/null) |"
  echo "| neuronx-cc / NxDI / vllm-neuron / torch-neuronx | $(ver neuronx-cc) / $(ver neuronx-distributed-inference) / $(ver vllm-neuron) / $(ver torch-neuronx) |"
  echo "| Runtime / driver | $(grep -m1 aws-neuronx-runtime-lib "$RUN/inventory.log" 2>/dev/null | awk '{print $3}') / $(grep -m1 aws-neuronx-dkms "$RUN/inventory.log" 2>/dev/null | awk '{print $3}') |"
  echo "| Updated | $(date -u +%FT%TZ) |"
  echo
  echo "## Config"
  echo '```'
  grep -vE '^\s*(#|$)' "$RUN/config.env" 2>/dev/null
  echo '```'
  echo
  echo "## Steps"
  echo "| Step | Exit | Duration | Note |"
  echo "|---|---|---|---|"
  while read -r name rc dur rest; do
    case "$name" in SKIP|No) echo "| ${name} | - | - | $rc $dur $rest |"; continue ;; esac
    note=""
    if [ "${rc#rc=}" != 0 ]; then   # last error-looking line of the step log
      note=$(grep -aE '(Error|Exception):|Unsupported|FAILED' "$RUN/$name.log" 2>/dev/null |
             tail -1 | sed 's/\x1b\[[0-9;]*m//g; s/|/\\|/g' | cut -c1-200)
    fi
    case "$name" in compile_*)   # cache hit/miss, compile time, peak host memory
      note="$(grep -ahE '^(cache hit|cache miss|compile_seconds|uploaded)' "$RUN/$name.log" 2>/dev/null | tr '\n' ' ')$(grep -a '^peak_host_mem' "$RUN/$name.log" 2>/dev/null | tail -1) $note" ;;
    esac
    echo "| $name | ${rc#rc=} | $dur | $note |"
  done < "$RUN/steps.txt"
  if ls "$RUN"/bench_*/benchmark_report.json >/dev/null 2>&1; then
    echo
    echo "## NxDI latency (p50 ms)"
    echo "| TP | Batch | Prefill | Decode/step | E2E | E2E tok/s |"
    echo "|---|---|---|---|---|---|"
    for r in "$RUN"/bench_*/benchmark_report.json; do
      python3 - "$r" <<'EOF'
import json, os, re, sys
d = json.load(open(sys.argv[1]))
m = re.match(r"bench_(?:tp(\d+)_)?b(\d+)$", os.path.basename(os.path.dirname(sys.argv[1])))
g = lambda k, f: d.get(k, {}).get(f, float("nan"))
print(f"| {m.group(1) or '?'} | {m.group(2)} | {g('context_encoding_model','latency_ms_p50'):.2f} "
      f"| {g('token_generation_model','latency_ms_p50'):.2f} "
      f"| {g('e2e_model','latency_ms_p50'):.0f} | {g('e2e_model','throughput'):.0f} |")
EOF
    done
  fi
  if ls "$RUN"/vllm_*.json >/dev/null 2>&1; then
    echo
    echo "## vLLM serving"
    echo "| Result | TTFT p50 / p99 ms | TPOT p50 ms | ITL p50 ms | Output tok/s | Req/s |"
    echo "|---|---|---|---|---|---|"
    for r in "$RUN"/vllm_*.json; do
      python3 - "$r" <<'EOF'
import json, os, sys
d = json.load(open(sys.argv[1])); g = lambda k: d.get(k, float("nan"))
print(f"| {os.path.basename(sys.argv[1])[:-5]} | {g('median_ttft_ms'):.1f} / {g('p99_ttft_ms'):.1f} "
      f"| {g('median_tpot_ms'):.2f} | {g('median_itl_ms'):.2f} | {g('output_throughput'):.0f} | {g('request_throughput'):.2f} |")
EOF
    done
  fi
  for f in "$RUN"/profile_files*.txt; do
    [ -s "$f" ] || continue
    echo
    echo "## $(basename "$f" .txt) (bytes, path)"
    echo '```'; cat "$f"; echo '```'
  done
} > "$RUN/RUN.md"
