import os
os.chdir("/Users/saksham/amazon/neuron/neuron-llm-profiling")


def edit(p, pairs):
    s = open(p).read()
    for old, new in pairs:
        assert old in s, (p, old[:70])
        s = s.replace(old, new)
    open(p, "w").write(s)


edit("bench/config.env", [
("""TP=2                 # trn1.2xlarge = 1 Trainium chip = 2 NeuronCore-v2
BATCHES="1 4"        # each batch size is a separate compile (~5-10 min each on 8 vCPUs)""",
"""TPS="1 2"            # tensor-parallel degrees to compare; trn1.2xlarge = 1 chip = 2 NeuronCore-v2
BATCHES="1 4"        # every TP x batch pair is a separate compile (~3 min each on trn1.2xlarge)"""),
("""# Serving benchmark (vLLM + vllm-neuron), best effort: skipped if no vLLM venv or <20 min left""",
"""# Serving benchmark (vLLM + vllm-neuron): one server at VLLM_TP, one client run per concurrency.
# Skipped if no vLLM venv or <20 min left.
VLLM_TP=2"""),
])

edit("bench/run.sh", [
("""#   3 bench_bN      NxD Inference compile + latency benchmark per batch size (prefill vs decode)
#   4 profile_b1    re-run batch 1 with system + device profiling, summarize per NEFF
#   5 vllm_cN       serving benchmark (TTFT / ITL / throughput) per concurrency""",
"""#   3 bench_tpT_bN  NxD Inference compile + latency benchmark per TP degree x batch size (prefill vs decode)
#   4 profile_tpT   re-run batch 1 per TP degree with system + device profiling, summarize per NEFF
#   5 vllm_tpT      serving benchmark (TTFT / TPOT / ITL / throughput), one server, each concurrency"""),
("""demo() {  # demo <batch> [extra args...]
  local b=$1; shift
  "$NXDI_VENV/bin/inference_demo" --model-type "$MODEL_TYPE" --task-type causal-lm run \\
    --model-path "$MODEL_DIR" --compiled-model-path "$HOME/compiled/$(basename "$MODEL_ID")-tp$TP-b$b" \\
    --torch-dtype bfloat16 --tp-degree "$TP" --batch-size "$b" \\""",
"""demo() {  # demo <tp> <batch> [extra args...]
  local tp=$1 b=$2; shift 2
  "$NXDI_VENV/bin/inference_demo" --model-type "$MODEL_TYPE" --task-type causal-lm run \\
    --model-path "$MODEL_DIR" --compiled-model-path "$HOME/compiled/$(basename "$MODEL_ID")-tp$tp-b$b" \\
    --torch-dtype bfloat16 --tp-degree "$tp" --batch-size "$b" \\"""),
("""bench() {  # inference_demo writes benchmark_report.json to cwd
  local b=$1 out=$PWD/$RUN/bench_b$1
  mkdir -p "$out" && cd "$out" && demo "$b" --benchmark
}""",
"""bench() {  # bench <tp> <batch>; inference_demo writes benchmark_report.json to cwd
  local out=$PWD/$RUN/bench_tp$1_b$2
  mkdir -p "$out" && cd "$out" && demo "$1" "$2" --benchmark
}"""),
("""profile() {
  mkdir -p "$PROFILE_DIR\"""",
"""profile() {  # profile <tp>: batch 1, reuses the compiled model from bench
  local tp=$1 dir=$PROFILE_DIR/tp$1
  mkdir -p "$dir\""""),
("""  NEURON_RT_INSPECT_DEVICE_PROFILE=1 NEURON_RT_INSPECT_OUTPUT_DIR="$PROFILE_DIR" \\
    demo 1 --skip-compile || demo 1   # older NxDI: no --skip-compile, reuses compiled path
  local n; n=$(find "$PROFILE_DIR" -name '*.ntff' | wc -l)""",
"""  NEURON_RT_INSPECT_DEVICE_PROFILE=1 NEURON_RT_INSPECT_OUTPUT_DIR="$dir" \\
    demo "$tp" 1 --skip-compile || demo "$tp" 1   # older NxDI: no --skip-compile, reuses compiled path
  local n; n=$(find "$dir" -name '*.ntff' | wc -l)"""),
("""  "$tool" view -d "$PROFILE_DIR" --output-format summary-text > "$RUN/profile_summary.txt" 2>&1
  find "$PROFILE_DIR" \\( -name '*.neff' -o -name '*.ntff' \\) -printf '%10s  %p\\n' | sort -k2 > "$RUN/profile_files.txt"   # bytes  path
  du -sh "$PROFILE_DIR\"""",
"""  "$tool" view -d "$dir" --output-format summary-text > "$RUN/profile_summary_tp$tp.txt" 2>&1
  find "$dir" \\( -name '*.neff' -o -name '*.ntff' \\) -printf '%10s  %p\\n' | sort -k2 > "$RUN/profile_files_tp$tp.txt"   # bytes  path
  du -sh "$dir\""""),
("""  mkdir -p "$RUN/profile_raw"
  find "$PROFILE_DIR" \\( -name '*.neff' -o -name '*.ntff' \\) -exec cp {} "$RUN/profile_raw/" \\;
  if [ -n "$S3_BUCKET" ]; then aws s3 sync "$PROFILE_DIR" "$S3_BUCKET/$(basename "$RUN")/"; fi""",
"""  mkdir -p "$RUN/profile_raw/tp$tp"
  find "$dir" \\( -name '*.neff' -o -name '*.ntff' \\) -exec cp {} "$RUN/profile_raw/tp$tp/" \\;
  if [ -n "$S3_BUCKET" ]; then aws s3 sync "$dir" "$S3_BUCKET/$(basename "$RUN")/tp$tp/"; fi"""),
("""vllm_bench() {
  local c=$1 port=8000
  "$VLLM_VENV/bin/vllm" serve "$MODEL_DIR" --tensor-parallel-size "$TP" --max-num-seqs 4 \\
    --max-model-len "$SEQ_LEN" --no-enable-prefix-caching --port $port > "$RUN/vllm_server_c$c.log" 2>&1 &""",
"""vllm_bench() {  # vllm_bench <tp>: one server (compile once), then one client run per concurrency
  local tp=$1 port=8000 log=$RUN/vllm_server_tp$1.log rc=0
  # Flags per the Neuron 2.28 vLLM quickstart; prefix caching needs block-KV config we don't set up.
  VLLM_NEURON_FRAMEWORK=neuronx-distributed-inference NEURON_COMPILED_ARTIFACTS=$HOME/compiled/vllm-tp$tp \\
  "$VLLM_VENV/bin/vllm" serve "$MODEL_DIR" --tensor-parallel-size "$tp" --max-num-seqs 4 \\
    --max-model-len "$SEQ_LEN" --no-enable-prefix-caching --port $port > "$log" 2>&1 &"""),
("""    echo "vLLM server not healthy; last log lines:"; tail -20 "$RUN/vllm_server_c$c.log\"""",
"""    echo "vLLM server not healthy; last log lines:"; tail -20 "$log\""""),
("""  "$VLLM_VENV/bin/vllm" bench serve --model "$MODEL_DIR" --port $port \\
    --dataset-name random --random-input-len "$VLLM_INPUT_LEN" --random-output-len "$VLLM_OUTPUT_LEN" \\
    --num-prompts "$VLLM_NUM_PROMPTS" --max-concurrency "$c" \\
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \\
    --save-result --result-dir "$RUN" --result-filename "vllm_c$c.json"
  kill $pid; wait $pid""",
"""  for c in $VLLM_CONCURRENCY; do
    echo "== concurrency $c"
    "$VLLM_VENV/bin/vllm" bench serve --model "$MODEL_DIR" --port $port \\
      --dataset-name random --random-input-len "$VLLM_INPUT_LEN" --random-output-len "$VLLM_OUTPUT_LEN" \\
      --num-prompts "$VLLM_NUM_PROMPTS" --max-concurrency "$c" \\
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \\
      --save-result --result-dir "$RUN" --result-filename "vllm_tp${tp}_c$c.json" || rc=1
  done
  kill $pid; wait $pid
  return $rc"""),
("""  for b in $BATCHES; do step "bench_b$b" 12 bench "$b"; done
  step profile_b1 8 profile
fi
if [ -n "$VLLM_VENV" ]; then
  for c in $VLLM_CONCURRENCY; do step "vllm_c$c" 20 vllm_bench "$c"; done
fi""",
"""  for tp in $TPS; do for b in $BATCHES; do step "bench_tp${tp}_b$b" 8 bench "$tp" "$b"; done; done
  for tp in $TPS; do step "profile_tp$tp" 6 profile "$tp"; done
fi
if [ -n "$VLLM_VENV" ]; then
  step "vllm_tp$VLLM_TP" 20 vllm_bench "$VLLM_TP"
fi"""),
])

edit("bench/roofline.py", [
("import json\n", "import json\nimport re\n"),
("""for report in sorted(glob.glob(os.path.join(run_dir, "bench_b*", "benchmark_report.json"))):
    batch = int(os.path.basename(os.path.dirname(report))[len("bench_b"):])
    prefill_floor_b_ms = prefill_floor_ms * batch  # batch x max_ctx tokens of prefill FLOPs
    print(f"## {os.path.relpath(report, run_dir)}  (batch {batch})")""",
"""for report in sorted(glob.glob(os.path.join(run_dir, "bench_*", "benchmark_report.json"))):
    m = re.match(r"bench_(?:tp(\\d+)_)?b(\\d+)$", os.path.basename(os.path.dirname(report)))
    tp, batch = m.group(1) or "?", int(m.group(2))
    prefill_floor_b_ms = prefill_floor_ms * batch  # batch x max_ctx tokens of prefill FLOPs
    print(f"## {os.path.relpath(report, run_dir)}  (TP {tp}, batch {batch})")"""),
])

edit("bench/runcard.sh", [
("""  if ls "$RUN"/bench_b*/benchmark_report.json >/dev/null 2>&1; then
    echo
    echo "## Latency (p50 ms)"
    echo "| Batch | Prefill | Decode/step | E2E | E2E tok/s |"
    echo "|---|---|---|---|---|"
    for r in "$RUN"/bench_b*/benchmark_report.json; do
      python3 - "$r" <<'EOF'
import json, os, sys
d = json.load(open(sys.argv[1])); b = os.path.basename(os.path.dirname(sys.argv[1]))[7:]
g = lambda k, f: d.get(k, {}).get(f, float("nan"))
print(f"| {b} | {g('context_encoding_model','latency_ms_p50'):.2f} | {g('token_generation_model','latency_ms_p50'):.2f} "
      f"| {g('e2e_model','latency_ms_p50'):.0f} | {g('e2e_model','throughput'):.0f} |")
EOF
    done
  fi""",
"""  if ls "$RUN"/bench_*/benchmark_report.json >/dev/null 2>&1; then
    echo
    echo "## NxDI latency (p50 ms)"
    echo "| TP | Batch | Prefill | Decode/step | E2E | E2E tok/s |"
    echo "|---|---|---|---|---|---|"
    for r in "$RUN"/bench_*/benchmark_report.json; do
      python3 - "$r" <<'EOF'
import json, os, re, sys
d = json.load(open(sys.argv[1]))
m = re.match(r"bench_(?:tp(\\d+)_)?b(\\d+)$", os.path.basename(os.path.dirname(sys.argv[1])))
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
  fi"""),
("""  if [ -s "$RUN/profile_files.txt" ]; then
    echo
    echo "## Profile files (bytes, path)"
    echo '```'; cat "$RUN/profile_files.txt"; echo '```'
  fi""",
"""  for f in "$RUN"/profile_files*.txt; do
    [ -s "$f" ] || continue
    echo
    echo "## $(basename "$f" .txt) (bytes, path)"
    echo '```'; cat "$f"; echo '```'
  done"""),
])
print("edits applied")
