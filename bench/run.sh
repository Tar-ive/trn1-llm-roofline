#!/usr/bin/env bash
# On-instance pipeline. Each step logs to results/<run>/<step>.log and is pushed to git
# right after it finishes, so a timeout loses at most the step in progress.
#
# MODE=run (Trainium box)                      MODE=compile (CPU box, targets trn1)
#   inventory, swap, model                       inventory, swap, model
#   compile_tpT_bN  S3 cache hit or compile      compile_tpT_bN  compile + upload to S3, then power off
#   bench_tpT_bN    latency: prefill vs decode
#   profile_tpT     system + device traces
#   vllm_tpT        serving: TTFT / TPOT / ITL / throughput
#   roofline, plots measured vs hardware floor
#
# Compiled models are cached in s3://<bucket>/compiled/<model>/<variant>/. The variant name hashes
# everything baked into the NEFFs (model, TP, batch, lengths, flags, compiler + NxDI versions), so a
# changed knob is a cache miss, never a stale hit.
set -uo pipefail
cd "$(dirname "$0")/.."
source bench/config.env
[ -f /etc/neuron-lab.env ] && source /etc/neuron-lab.env   # written by user-data: mode + bucket
MODE=${NEURON_LAB_MODE:-run}
BUCKET=${NEURON_LAB_BUCKET:-}
[ -n "$BUCKET" ] && S3_BUCKET=${S3_BUCKET:-s3://$BUCKET/profiles}
# On a CPU-only box, compile for trn1 (must be set before neuronx_distributed is imported).
[ "$MODE" = compile ] && export NEURON_PLATFORM_TARGET_OVERRIDE=trn1

RUN=results/$(date -u +%Y%m%dT%H%M%SZ)-$MODE
mkdir -p "$RUN"; ln -sfn "$(basename "$RUN")" results/latest
exec > >(tee -a "$RUN/console.log") 2>&1
cp bench/config.env "$RUN/"

DEADLINE=$(cat /etc/neuron-lab-deadline 2>/dev/null || echo $(( $(date +%s) + 86400 )))
mins_left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }
save() { /usr/local/bin/autosave "$1" || git add -A >/dev/null; }

step() {  # step <name> <min-minutes-needed> <cmd...>
  local name=$1 need=$2; shift 2
  if [ "$(mins_left)" -lt "$need" ]; then
    echo "SKIP $name ($(mins_left) min left, needs $need)" | tee -a "$RUN/steps.txt"; return 1
  fi
  echo "== $name ($(mins_left) min left)"
  local t0=$(date +%s)
  ( "$@" ) > "$RUN/$name.log" 2>&1
  local rc=$?
  echo "$name rc=$rc $(( $(date +%s) - t0 ))s" | tee -a "$RUN/steps.txt"
  bench/runcard.sh "$RUN"
  save "$name rc=$rc"
  return $rc
}

# Neuron venvs on the DLAMI are versioned (/opt/aws_neuronx_venv_*); find the one that can import
# a module, preferring venvs whose name matches $2 (e.g. the dedicated *nxd_inference* venv).
# Never pip-install NxDI on top: mismatched torch/torch-xla wheels break the venv (undefined _XLAC symbols).
find_venv() {
  for v in $(ls -d /opt/aws_neuronx_venv_*$2* 2>/dev/null) $(ls -d /opt/aws_neuronx_venv_* 2>/dev/null); do
    timeout 300 "$v/bin/python" -c "import $1" >/dev/null 2>&1 && { echo "$v"; return; }
  done
}
NXDI_VENV=$(find_venv neuronx_distributed_inference nxd_inference)
VLLM_VENV=$(find_venv vllm_neuron vllm)
PY_VENV=${NXDI_VENV:-$(ls -d /opt/aws_neuronx_venv_* | head -1)}
# torch_neuronx shells out to helpers in the venv's bin/ (e.g. libneuronpjrt-path), so it must be on PATH.
export PATH="${NXDI_VENV:+$NXDI_VENV/bin:}${VLLM_VENV:+$VLLM_VENV/bin:}$PATH"
echo "NXDI_VENV=${NXDI_VENV:-none} VLLM_VENV=${VLLM_VENV:-none} MODE=$MODE BUCKET=${BUCKET:-none}" | tee "$RUN/venvs.txt"
[ -n "$NXDI_VENV" ] || echo "No venv with NxD Inference on this AMI. On trn1/inf2 use a Neuron 2.28 DLAMI (see infra/launch.sh)." | tee -a "$RUN/steps.txt"

pkgver() { "$PY_VENV/bin/python" -c "import importlib.metadata as m; print(m.version('$1'))" 2>/dev/null || echo none; }
SDK="cc$(pkgver neuronx-cc)-nxdi$(pkgver neuronx-distributed-inference)"
MODEL_NAME=$(basename "$MODEL_ID")
MODEL_DIR=$HOME/models/$MODEL_NAME
PROFILE_DIR=$HOME/profiles/$(basename "$RUN")
export XLA_HLO_DEBUG=1 XLA_IR_DEBUG=1   # keep framework op/layer names in the NEFF so profiles map back to code

# Everything below is baked into the compiled model; it all goes into the cache key.
FIXED_ARGS="--torch-dtype bfloat16 --on-device-sampling --top-k 1 --pad-token-id $PAD_ID"
variant_key() { echo "model=$MODEL_ID type=$MODEL_TYPE tp=$1 batch=$2 ctx=$MAX_CTX seq=$SEQ_LEN args=$FIXED_ARGS extra=${NXDI_EXTRA:-} sdk=$SDK target=trn1"; }
variant() { echo "tp$1-b$2-ctx$MAX_CTX-seq$SEQ_LEN-$(variant_key "$1" "$2" | sha256sum | cut -c1-10)"; }
cdir()  { echo "$HOME/compiled/$MODEL_NAME/$(variant "$1" "$2")"; }
s3dir() { echo "s3://$BUCKET/compiled/$MODEL_NAME/$(variant "$1" "$2")"; }

demo() {  # demo <tp> <batch> [extra args...]
  local tp=$1 b=$2; shift 2
  # shellcheck disable=SC2086  # FIXED_ARGS / NXDI_EXTRA are flag lists
  "$NXDI_VENV/bin/inference_demo" --model-type "$MODEL_TYPE" --task-type causal-lm run \
    --model-path "$MODEL_DIR" --compiled-model-path "$(cdir "$tp" "$b")" \
    --tp-degree "$tp" --batch-size "$b" --max-context-length "$MAX_CTX" --seq-len "$SEQ_LEN" \
    $FIXED_ARGS --prompt "$PROMPT" ${NXDI_EXTRA:-} "$@"
}

mem_watch() {  # prints each new peak of host RAM + swap use; the last line is the run's peak
  local peak=0 ram swp
  while sleep 5; do
    read -r ram swp < <(free -m | awk '/^Mem:/{r=$3} /^Swap:/{s=$3} END{print r, s}')
    [ $((ram + swp)) -gt $peak ] && { peak=$((ram + swp)); echo "peak_host_mem_mb=$peak (ram $ram + swap $swp)"; }
  done
}

compile() {  # compile <tp> <batch>: local copy, else S3 cache, else compile (and upload)
  local tp=$1 b=$2 dir s3; dir=$(cdir "$tp" "$b"); s3=$(s3dir "$tp" "$b")
  echo "variant $(variant "$tp" "$b"): $(variant_key "$tp" "$b")"
  if [ -f "$dir/manifest.txt" ]; then echo "cache hit (local) $dir"; return 0; fi
  if [ -n "$BUCKET" ] && aws s3 ls "$s3/manifest.txt" >/dev/null 2>&1; then
    local t0=$(date +%s)
    aws s3 sync --only-show-errors "$s3" "$dir" || return 1
    echo "cache hit (s3) $s3: $(du -sh "$dir" | cut -f1) in $(( $(date +%s) - t0 ))s"; return 0
  fi
  echo "cache miss: compiling"
  mem_watch & local mw=$! t0=$(date +%s)
  demo "$tp" "$b" --compile-only; local rc=$?
  kill $mw 2>/dev/null; wait $mw 2>/dev/null
  echo "compile_seconds=$(( $(date +%s) - t0 )) rc=$rc"
  [ $rc = 0 ] || return $rc
  variant_key "$tp" "$b" > "$dir/manifest.txt"
  du -sh "$dir"; find "$dir" -name '*.neff' -printf '%10s  %P\n'
  if [ -n "$BUCKET" ]; then   # manifest goes last: its presence marks a complete upload
    t0=$(date +%s)
    aws s3 sync --only-show-errors "$dir" "$s3" --exclude manifest.txt &&
      aws s3 cp --only-show-errors "$dir/manifest.txt" "$s3/manifest.txt" &&
      echo "uploaded to $s3 in $(( $(date +%s) - t0 ))s"
  fi
}

bench() {  # bench <tp> <batch>; inference_demo writes benchmark_report.json to cwd
  local out=$PWD/$RUN/bench_tp$1_b$2
  mkdir -p "$out" && cd "$out" && demo "$1" "$2" --skip-compile --benchmark
}

profile() {  # profile <tp>: batch 1, reuses the compiled model
  local tp=$1 dir=$PROFILE_DIR/tp$1
  mkdir -p "$dir"
  # model mode: device trace for the first execution of each unique NEFF per core,
  # i.e. one trace for context encoding (prefill) and one per token-gen bucket (decode)
  NEURON_RT_INSPECT_ENABLE=1 NEURON_RT_INSPECT_SYSTEM_PROFILE=1 \
  NEURON_RT_INSPECT_DEVICE_PROFILE=1 NEURON_RT_INSPECT_OUTPUT_DIR="$dir" \
    demo "$tp" 1 --skip-compile
  local n; n=$(find "$dir" -name '*.ntff' | wc -l)
  [ "$n" -gt 0 ] || { echo "no device traces (.ntff) captured"; return 1; }
  local tool; tool=$(command -v neuron-explorer || command -v neuron-profile)
  "$tool" view -d "$dir" --output-format summary-text > "$RUN/profile_summary_tp$tp.txt" 2>&1
  find "$dir" \( -name '*.neff' -o -name '*.ntff' \) -printf '%10s  %p\n' | sort -k2 > "$RUN/profile_files_tp$tp.txt"   # bytes  path
  du -sh "$dir"
  # Raw traces go into git with the run; autosave skips (and lists in .git/info/exclude) any file >50MB.
  mkdir -p "$RUN/profile_raw/tp$tp"
  find "$dir" \( -name '*.neff' -o -name '*.ntff' \) -exec cp {} "$RUN/profile_raw/tp$tp/" \;
  if [ -n "$S3_BUCKET" ]; then aws s3 sync --only-show-errors "$dir" "$S3_BUCKET/$(basename "$RUN")/tp$tp/" && echo "traces: $S3_BUCKET/$(basename "$RUN")/tp$tp/"; fi
}

vllm_bench() {  # vllm_bench <tp>: one server (compile once, cached in S3), then one client run per concurrency
  local tp=$1 port=8000 log=$RUN/vllm_server_tp$1.log rc=0
  local key="model=$MODEL_ID tp=$tp seqs=${VLLM_MAX_SEQS:-4} len=$SEQ_LEN sdk=$SDK vllm-neuron=$(pkgver vllm-neuron)"
  local v; v="tp$tp-seqs${VLLM_MAX_SEQS:-4}-len$SEQ_LEN-$(echo "$key" | sha256sum | cut -c1-10)"
  local art=$HOME/compiled/vllm/$MODEL_NAME/$v s3=s3://$BUCKET/vllm/$MODEL_NAME/$v hit=0
  if [ -n "$BUCKET" ] && aws s3 ls "$s3/manifest.txt" >/dev/null 2>&1; then
    aws s3 sync --only-show-errors "$s3" "$art" && hit=1 && echo "vLLM artifacts: cache hit $s3"
  fi
  mkdir -p "$art"
  # Flags per the Neuron 2.28 vLLM quickstart; prefix caching needs block-KV config we don't set up.
  VLLM_NEURON_FRAMEWORK=neuronx-distributed-inference NEURON_COMPILED_ARTIFACTS=$art \
  "$VLLM_VENV/bin/vllm" serve "$MODEL_DIR" --tensor-parallel-size "$tp" --max-num-seqs "${VLLM_MAX_SEQS:-4}" \
    --max-model-len "$SEQ_LEN" --no-enable-prefix-caching --port $port > "$log" 2>&1 &
  local pid=$! up=0 t0=$(date +%s)
  for _ in $(seq $(( ${VLLM_MIN:-20} * 12 ))); do   # up to VLLM_MIN min for compile + load; bail out as soon as the server dies
    curl -fs localhost:$port/health >/dev/null && { up=1; break; }
    kill -0 $pid 2>/dev/null || break
    sleep 5
  done
  if [ $up = 0 ]; then
    echo "vLLM server not healthy; last log lines:"; tail -20 "$log"
    kill $pid 2>/dev/null; return 1
  fi
  echo "vLLM server ready in $(( $(date +%s) - t0 ))s (cache hit=$hit)"
  if [ -n "$BUCKET" ] && [ $hit = 0 ] && [ -n "$(ls -A "$art")" ]; then
    echo "$key" > "$art/manifest.txt"
    aws s3 sync --only-show-errors "$art" "$s3" --exclude manifest.txt &&
      aws s3 cp --only-show-errors "$art/manifest.txt" "$s3/manifest.txt" && echo "vLLM artifacts uploaded to $s3"
  fi
  for c in $VLLM_CONCURRENCY; do
    echo "== concurrency $c"
    "$VLLM_VENV/bin/vllm" bench serve --model "$MODEL_DIR" --port $port \
      --dataset-name random --random-input-len "$VLLM_INPUT_LEN" --random-output-len "$VLLM_OUTPUT_LEN" \
      --num-prompts "$VLLM_NUM_PROMPTS" --max-concurrency "$c" \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
      --save-result --result-dir "$RUN" --result-filename "vllm_tp${tp}_c$c.json" || rc=1
  done
  kill $pid; wait $pid
  return $rc
}

download() {
  local cli=$PY_VENV/bin/hf; [ -x "$cli" ] || cli=$PY_VENV/bin/huggingface-cli
  "$cli" download "$MODEL_ID" --local-dir "$MODEL_DIR" ${DOWNLOAD_EXCLUDE:+--exclude "$DOWNLOAD_EXCLUDE"}
  du -sh "$MODEL_DIR"
}

swap() {  # host RAM is 30 GiB on trn1.2xlarge; an 8B checkpoint is 16 GB before sharding
  [ "${SWAP_GB:-0}" -gt 0 ] || return 0
  swapon --show | grep -q /swapfile || {
    sudo fallocate -l "${SWAP_GB}G" /swapfile && sudo chmod 600 /swapfile &&
    sudo mkswap /swapfile && sudo swapon /swapfile; }
  free -g
}

step inventory 0 bench/inventory.sh
step swap 0 swap
step model 0 download
if [ -n "$NXDI_VENV" ]; then
  "$NXDI_VENV/bin/inference_demo" --model-type "$MODEL_TYPE" --task-type causal-lm run --help > "$RUN/inference_demo_help.txt" 2>&1
  for tp in $TPS; do for b in $BATCHES; do
    step "compile_tp${tp}_b$b" "${COMPILE_MIN:-8}" compile "$tp" "$b" &&
      [ "$MODE" = run ] && step "bench_tp${tp}_b$b" 5 bench "$tp" "$b"
  done; done
  if [ "$MODE" = run ]; then
    for tp in $TPS; do step "profile_tp$tp" 6 profile "$tp"; done
  fi
fi
if [ "$MODE" = run ] && [ -n "$VLLM_VENV" ] && [ "${VLLM_ENABLE:-1}" = 1 ]; then
  step "vllm_tp$VLLM_TP" "${VLLM_MIN:-20}" vllm_bench "$VLLM_TP"
fi
if [ "$MODE" = run ]; then
  step roofline 0 "$PY_VENV/bin/python" bench/roofline.py "$RUN" "$MODEL_DIR"
  step plots 0 "$PY_VENV/bin/python" bench/plots.py "$RUN"
fi
echo "done with $(mins_left) min left"
bench/runcard.sh "$RUN"
save "run complete $(basename "$RUN")

$(cat "$RUN/steps.txt")"
# Don't pay for idle time: power off (= terminate; the shutdown hook pushes once more).
if [ "${POWEROFF_WHEN_DONE:-1}" = 1 ] && [ -f /etc/neuron-lab-deadline ]; then sudo systemctl poweroff; fi
