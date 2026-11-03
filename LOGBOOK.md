# Logbook

What was tried, what broke, and what changed. Newest first. One entry per launch attempt.
Each run also has a generated card at `results/<run>/RUN.md` (stack versions, step exit codes,
latency table, profile files). This file holds the narrative: challenges, root causes, fixes, next steps.

Entry template:

```
## YYYY-MM-DD run <results dir or "none"> (instance, AMI/SDK, ~cost)
Goal:
What happened:
Root cause:
Fix (commit):
Results:
Next:
```

---

## 2026-10-31 run 20261031T032009Z-run (i-097f1ef830577fb23, trn1.2xlarge us-east-1f, Neuron 2.28 DLAMI, ~$1.40)

**Goal:** first Llama-3.1-8B bf16 baseline on trn1 using the S3 compile cache.

**What happened**
- us-east-1c: InsufficientInstanceCapacity; launched in us-east-1f.
- Both compiles were S3 cache hits: 15-16 GB in ~90 s each (vs 6-9 min to compile on 16 vCPUs). No compile on the trn1.
- Benchmarks took much longer than budgeted: b1 779 s, b16 1091 s. NxDI's benchmark runs 20 iterations
  (`num_runs=20`, not a CLI flag in 0.8) of a full 1024-token generation, 32-46 s each.
- `profile_tp2` started ~04:01 and was still running when the 04:14 final-save timer powered the box off.
  `neuron-profile view --output-format summary-text` on the 8B traces is slow; no summary, no rc, and the raw traces
  never reached S3 (the sync is the last thing the step does). vLLM, roofline and plots never ran.
- Rescued by hand over SSH: the decode NEFF trace (core 0) converted to Perfetto (`neuron-profile view
  --output-format perfetto`, ~3.5 min and ~630 MB per trace) and copied to `traces/` (gitignored). Opened in Chrome
  via ui.perfetto.dev's postMessage API from a local page (`traces/open.html`); `?url=` did not fetch.
- No `final:` commit: the last push from the box was the 04:10 periodic one.
- roofline.log and plots generated locally afterwards (weights 16.06 GB from the HF file sizes).

**Results** (TP=2, 1024-token prefill, 2048 seq, p50; p99 within 1%)

| Batch | Prefill | Decode/step | Decode tok/s | E2E tok/s | Prefill % of BF16 peak | Decode % of HBM BW (weights only) |
|---|---|---|---|---|---|---|
| 1 | 146.9 ms | 31.29 ms | 32 | 63 | 59% | 63% |
| 16 | 2310.8 ms | 42.06 ms | 381 | 712 | 60% | 47% |

- vs TinyLlama: prefill 34-39% -> 59-60%, decode 51-55% -> 63% at b1. Fixed costs amortize over a bigger model.
- Prefill plateaus at ~60% of peak from b1 to b16: compute-bound with a constant ~40% gap. NeuronMLP's NKI matmul
  baseline reaches 64% MFU, so this is close to what un-fused kernels get.
- Decode b16: 12x the tokens/s for 34% more step time. The 47% understates efficiency: roofline.py ignores KV-cache
  reads, ~3.2 GB/step at b16 (16 seqs x ~1536 avg tokens x 128 KB/token) = +3.9 ms, floor ~23.5 ms, so ~56%.
- Perfetto decode trace (core 0, 31.05 ms): six DMA queues busy for the whole step: weight streaming dominates.

**Next**
- Budget: benches alone are ~31 min. Either raise LAUNCH_MINUTES to ~100, or split runs (bench+profile / vLLM).
- profile step: give it its own time, convert prefill and decode traces to Perfetto on the box, upload to S3 *before*
  the slow summary-text pass.
- roofline.py: add KV-cache bytes to the decode floor.
- vLLM with realistic traffic (Poisson arrivals, ShareGPT lengths, rate sweep) still to run.

---

## 2026-10-30 run 20261030T175623Z-compile (i-0520a98e34ea67d9f, m7i.4xlarge us-east-1b, Neuron 2.28 DLAMI, ~$0.35)

**Goal:** compile Llama-3.1-8B (TP=2, batch 1 and 16) on a CPU-only box for trn1 and fill the S3 cache.

**What happened**
- CPU compile works on Neuron 2.28 (`NEURON_PLATFORM_TARGET_OVERRIDE=trn1`, `inference_demo --compile-only`).
- b1: 364 s compile, peak host RAM 26.2 GB; b16: 553 s, 28.0 GB. Swap never used on the 64 GiB box,
  but 26-28 GB would have been tight on trn1.2xlarge (30 GiB). Weight sharding took 78 s of each compile.
- Uploads: 15-16 GB per variant in ~29 s. Cache now holds `tp2-b1-...-a5e23979ca` and `tp2-b16-...-949b2a9e8b` (32 GB).
- Finished with 97 of 120 min unused and powered off (POWEROFF_WHEN_DONE works). Teardown not run; the compile deploy key
  remains until the next compile launch removes it.

**Warning investigated:** every attention layer logs
`TP degree (2) and KV heads (8) are not divisible. Overriding attention sharding strategy to GQA.CONVERT_TO_MHA!`.
Read from the NxDI 0.8.16251 source (`modules/attention/gqa.py`):
- `determine_sharding_strategy` checks `tp_degree % kv_heads` (2 % 8 = 2). That test is for the case of more cores
  than KV heads (e.g. TP=32, 8 KV heads), where each KV head is copied to several cores. It fails here, so the label
  switches to CONVERT_TO_MHA.
- `get_shardable_head_counts` only changes the KV head count when `kv_heads < tp_degree` or `kv_heads % tp_degree != 0`.
  Neither holds (8 >= 2, 8 % 2 = 0), so KV heads stay 8, i.e. 4 per core. The KV cache manager uses the same function.
- Net effect: none. The warning is cosmetic for TP=2. KV cache at b16 x 2048 tokens is ~4 GB total (2 GB/core), as planned.

**Next:** 60-min trn1 run; all compile steps should be S3 cache hits.

---

## Applied in next runs: S3 compile cache, CPU compile mode, plots (Llama-3.1-8B)

**Why:** 8B compiles are the expensive, slow part (estimated 15-30 min per shape on trn1's 8 vCPUs) and every box
threw them away. The model will be used repeatedly, so compile once and reuse.

**Changes:**
- `infra/storage.sh`: private bucket `neuron-lab-<account>-<region>` + instance role/profile `neuron-lab-ec2`
  (S3 access to that bucket only). `launch.sh` runs it and attaches the profile.
- `MODE=compile ./infra/launch.sh`: m7i.4xlarge on the same 2.28 DLAMI, `NEURON_PLATFORM_TARGET_OVERRIDE=trn1`,
  `inference_demo --compile-only` per variant, upload, power off. Unverified that NxDI compiles without a device.
- `run.sh`: `compile_*` steps check local, then S3 (`manifest.txt` marks a complete upload), else compile and upload;
  log compile seconds and peak host RAM + swap. `bench_*` always uses `--skip-compile`. vLLM artifacts cached the same way.
  Raw traces default to `s3://<bucket>/profiles/`. `plots.py` writes PNG charts. Boxes power off when done.
- Budgets: trn1 run 60 min (was 150), compile box 120 min. Deploy keys and `.state/instance-<mode>.env` are per mode.

**Plan:** compile box first (b1, b16), then a 60-minute trn1 run that should be all cache hits.
Cache logic was dry-run locally with stubbed `aws`/`inference_demo` (miss + upload, local hit, S3 hit,
knob change = new variant, partial upload = miss).

---

## Applied in next run: Llama-3.1-8B bf16 (largest dense model that fits)

**Why:** TinyLlama (2.2 GB on a 32 GB chip, 2048-wide matmuls) is dominated by fixed costs (all-reduce, launch,
scheduling); batch 1 -> 4 added only 7% to the decode step. Kernel work tuned on it would target the wrong bottleneck.
8B bf16 is 16 GB: the ceiling for this chip with KV-cache headroom (14B bf16 ~28 GB doesn't fit).

**Changes:**
- `config.env`: `NousResearch/Meta-Llama-3.1-8B-Instruct` (ungated mirror, `llama` model type, skips `original/*`),
  TP=2 only, batch 1 and 16, 1024-token prefill / 2048 seq, `--save-sharded-checkpoint`, 32 GB swap,
  `COMPILE_MIN=30`; vLLM 512 in / 256 out, max 8 seqs, concurrency 1 and 8, `VLLM_MIN=35`; `LAUNCH_MINUTES=150`.
- `run.sh`: swap step, download excludes, `NXDI_EXTRA` passed to every `inference_demo`, per-step time
  thresholds and vLLM startup wait come from config.
- `launch.sh`: default `MINUTES` comes from `LAUNCH_MINUTES`.

**Expect:** decode floor 16 GB / 820 GB/s = 19.6 ms/step. TinyLlama reached 55% of HBM bandwidth; 8B should do better
because the fixed costs are amortized over 7x more bytes. Prefill should move up from 34-39% MFU with 4096-wide matmuls.
Watch: host RAM / swap during sharding, compile time per batch size, HBM fit at batch 16, vLLM vs NxDI TPOT gap.

**Risks:** compile time is a guess (TinyLlama: ~3 min; 8B could be 15-30 min per shape). If b16 compile or load OOMs,
drop to `BATCHES="1 8"`.

---

## 2026-10-29 run 20261029T003425Z (i-0422bf7dd5b82526a, trn1.2xlarge us-east-1f, Neuron 2.28 DLAMI, ~$1.34)

**Goal:** TP=1 vs TP=2, and verify the vLLM fix (one server, a client run per concurrency).

**What happened**
- All 10 steps rc=0; pipeline finished 01:03 UTC with ~30 min of budget unused.
- vLLM works: `--no-enable-prefix-caching` + `VLLM_NEURON_FRAMEWORK=neuronx-distributed-inference` got past the
  config error; one server served both concurrencies (338 s including compile).
- Raw traces lost again: they now land in `profile_raw/tp{1,2}/`, but `.gitignore` only un-ignored
  `profile_raw/*.ntff` (one level). Fixed to `profile_raw/**/*.ntff`. Only the summaries survived.
- us-east-1c had no capacity; launched in us-east-1f.
- SSH from the Mac timed out: the egress IP rotates per connection within <redacted-ip>. Allowed that /27.
  The box's `boot:` autosave pushes nothing (nothing to commit), so it can't serve as a ready signal.

**Results** (TinyLlama-1.1B bf16, 512-token prefill, p50)

| TP | Batch | Prefill | Decode/step | E2E tok/s |
|---|---|---|---|---|
| 1 | 1 | 24.9 ms | 9.26 ms | 207 |
| 1 | 4 | 118.3 ms | 9.93 ms | 754 |
| 2 | 1 | 17.5 ms | 4.92 ms | 374 |
| 2 | 4 | 60.3 ms | 5.26 ms | 1371 |

- TP=2 decode is 1.9x faster than TP=1. The all-reduce cost (7-31% per step) is far smaller than the gain from
  using both cores' HBM bandwidth. The TP=1 hypothesis was wrong.
- vLLM TP=2 (256 in / 128 out): c=1 TTFT 24 ms, TPOT 7.6 ms, 128 tok/s; c=4 TTFT 55 / 85 ms (p50 / p99),
  TPOT 7.8 ms, 490 tok/s. vLLM's per-token latency is ~55% above NxDI's raw decode step (7.6 vs 4.9 ms).

**Next**
- Re-run profiles to get raw traces (gitignore fixed); identify the unknown third graph.
- Explain the vLLM vs NxDI decode gap (scheduler/host overhead, bucket choice).
- Use the spare ~30 min: higher concurrency (8, 16) with `--max-num-seqs` raised, or longer context.

---

## Applied in run 20261029T003425Z: TP=1 vs TP=2 + vLLM fix

**Why:** the 20261027T222450Z profile showed decode at TP=2 losing 7-31% of each step to the cross-core
all-reduce, with core 1 consistently slower (6.3 vs 4.7 ms). A 1.1B model fits on one NeuronCore, so TP=1
may decode faster even with half the HBM bandwidth per core. Prefill should still favour TP=2.

**Changes** (applied via `plans/tp_sweep_and_vllm_edit.py`):
- `config.env`: `TP=2` becomes `TPS="1 2"`; add `VLLM_TP=2`.
- `run.sh`: benchmark every TP x batch pair (`bench_tp{1,2}_b{1,4}`, 4 compiles ~3 min each);
  profile batch 1 at each TP (`profile_tp{1,2}`, traces in `profile_raw/tp{1,2}/`).
- vLLM: one server per run instead of one per concurrency (compile once), client runs for each concurrency,
  results `vllm_tp2_c{1,4}.json`. Add `VLLM_NEURON_FRAMEWORK=neuronx-distributed-inference` and
  `NEURON_COMPILED_ARTIFACTS`, keep `--no-enable-prefix-caching` (all per the Neuron 2.28.1 vLLM online-serving quickstart).
- `roofline.py` and `runcard.sh`: parse `bench_tp{T}_b{B}` (old `bench_b{B}` still works), add a TP column and a vLLM
  table (TTFT p50/p99, TPOT, ITL, output tok/s, req/s).

**Compare:** decode ms/token and tok/s at TP=1 vs 2 (batch 1 and 4); prefill latency; collectives % and
per-core imbalance in the two profiles; whether vLLM now starts and what TTFT/TPOT it gets.

**Budget estimate:** ~5 min boot, ~13 min compiles+benchmarks, ~2 min profiles, ~10-15 min vLLM: fits in 55 min.

---

## 2026-10-27 run 20261027T222450Z (i-0de00ab5229d7de4f, trn1.2xlarge us-east-1f, Neuron 2.28 DLAMI, ~$1.26)

**Goal:** first real NxDI benchmark + device profile on Trn1, using the pinned Neuron 2.28 image.

**What happened**
- First import of torch/NxDI took ~2 min: a fresh EBS volume lazily loads blocks from its snapshot
  (`folio_wait_bit_common`). Expected on every new box; not a hang.
- `bench_b1`/`bench_b4` failed in 3 s: `FileNotFoundError: 'libneuronpjrt-path'`. run.sh called venv binaries
  by full path without putting the venv's `bin/` on `PATH`, and torch_neuronx shells out to a helper there.
  Fixed (c6bffbf), pushed mid-run, pulled on the box, pipeline restarted (aborted attempt: `20261027T221834Z`).
  While restarting, `pkill -f bench/run.sh` matched the SSH command itself and killed the session; use PIDs.
- After the fix: benchmarks, profile and roofline succeeded.
- vLLM (0.13 + vllm-neuron 0.4.1) failed at config time: `When prefix caching is enabled, block_size must be set`.
  Fix: `--no-enable-prefix-caching` (bd3878a). **Untested.**
- `roofline.py` reported batch-4 prefill at 10% of peak; it ignored batch size. Real value 39%. Fixed (bd3878a).
- The raw traces (9 files, 161 MB in `~/profiles`) were not in git and were lost at termination.
  Only `profile_summary.txt` survived. Now copied into `results/<run>/profile_raw/` (this commit).

**Results** (TinyLlama-1.1B bf16, TP=2, 512-token prefill, p50; p99 within 1%)

| | Batch 1 | Batch 4 | Floor | Efficiency |
|---|---|---|---|---|
| Prefill | 17.5 ms | 60.3 ms | 5.9 / 23.7 ms | 34% / 39% of BF16 peak |
| Decode / step | 4.92 ms | 5.26 ms | 2.68 ms | 55% / 51% of HBM BW |
| Decode tok/s | 203 | 762 | | 3.7x from batching |

Profile (per graph, % of graph time; phase mapping inferred from metrics):
- Decode: DMA busy 94% (core 0), MBU 55%, TensorE ~30%: memory-bound as expected.
  Core 1 spends 31% in collectives and is slower than core 0 (6.3 vs 4.7 ms): TP=2 all-reduce overhead.
- Prefill: TensorE active ~51-58%, MFU 33%, collectives ~13%, Vector+Scalar 20-34%, 26 MB SBUF spill.
- A third graph (10.4 ms, DMA 79%, no collectives, no VectorE) is unidentified; raw traces would have told us.

**Next**
- Verify vLLM serving with the prefix-caching fix.
- TP=1 vs TP=2 for decode (the model fits on one core; collectives cost 7-31% per step).
- Identify the third graph from the raw traces now kept in git.
- Longer context / bigger batch to see whether prefill MFU climbs.

---

## 2026-10-27 run 20261027T172741Z (i-076678c12cd22ad43, trn1.2xlarge us-east-1f, latest DLAMI = Neuron SDK 2.33, ~$1.26)

**Goal:** first full pipeline run on the lab VPC.

**What happened**
- SSH timed out: this Mac's public IP alternates (<redacted-ip> / <redacted-ip> / <redacted-ip> seen), and
  network.sh had reset the security group to a single IP. Fix: network.sh accumulates IPs in `.state/ssh_cidrs`.
- The latest DLAMI had no venv with NxD Inference. run.sh pip-installed NxDI 0.8 into the torch-neuronx 2.14 venv,
  which pulled torch-xla 2.13 (ABI mismatch: `_XLAC ... undefined symbol`). Every NxDI step failed.
- vllm-neuron 0.29: `ValueError: Unsupported instance family: trn1. Supported: trn2, trn3pd, trn3pds`.
- run.sh waited the full 15 min for a vLLM server that had already died, twice: half the hour wasted.
- The profile step reported rc=0 with no traces captured.

**Root cause:** Neuron 2.29+ dropped Trn1/Inf2 from NxD Inference (NKI 0.3 kernels); AWS says pin to 2.28.

**Fix (cfcee2c):** pin trn1/inf2 to "Deep Learning AMI Neuron PyTorch Inference vLLM 0.13 (Ubuntu 24.04) 20260227"
(Neuron 2.28, NxDI 0.8.16251, vllm-neuron 0.4.1); never pip-install NxDI; vLLM wait loop exits when the server dies;
profile step requires `.ntff` files; launch.sh refuses uncommitted changes and pulls before pushing.

**Shutdown:** clean. Final-save timer fired at 18:22:39, `Client.InstanceInitiatedShutdown`, terminate-on-shutdown.

---

## 2026-10-27 launch attempt, no run (VcpuLimitExceeded at quota 8)

Run 1's instance was still running and held all 8 vCPUs of the new quota.
**Fix (d43546a):** preflight subtracts vCPUs of running Trn instances and names them; quota vs capacity errors are reported separately.

---

## 2026-10-27 launch, no results (i-0d358dc9d10a6cd65, us-east-1f, ~$0.35, terminated by hand after ~15 min)

**What happened**
- us-east-1c had no trn1.2xlarge capacity; launch.sh fell through to us-east-1f.
- Setup died at `git clone`: the account's default VPC network ACL allowed inbound TCP 3306 only. NACLs are
  stateless, so replies to the box's outbound connections were dropped. SSH was blocked by the same ACL,
  and by a security group listing stale IPs.
- The deploy key was echoed into the EC2 console log by `set -x`.

**Fix (e2c4434):** dedicated `neuron-lab` VPC (network.sh, destroy-network.sh) so the shared default VPC
(EFS, ECS, another instance) is untouched; key written before `set -x`; stale `neuron-lab-*` deploy keys purged on launch.

---

## Before 2026-10-27: quota

First launches failed with `VcpuLimitExceeded`: "Running On-Demand Trn instances" (L-2C3B7624) was 0 in every
region that offers trn1.2xlarge (us-east-1, us-east-2, us-west-2), and spot was 0 too.
Requested 8 vCPUs in us-east-1 (approved). Added `infra/regions.sh` and the quota/AZ preflight (7818e70).

## Cost so far

| Instance | Minutes | Cost |
|---|---|---|
| i-0d358dc9d10a6cd65 | ~15 | ~$0.35 |
| i-076678c12cd22ad43 | ~56 | ~$1.26 |
| i-0de00ab5229d7de4f | ~56 | ~$1.26 |
| **Total** | | **~$2.87** |

trn1.2xlarge on-demand us-east-1: $1.34375/hr, per-second billing. EBS and public IPv4 add cents.
