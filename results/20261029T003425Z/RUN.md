# Run 20261029T003425Z

| | |
|---|---|
| Instance | i-0422bf7dd5b82526a (trn1.2xlarge, us-east-1f) |
| AMI | ami-00c45d97f3070117d |
| Repo commit | 07ff7a1 Apply TP=1 vs TP=2 sweep and single-server vLLM benchmark (one client run per concurrency) |
| Venvs | NXDI_VENV=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13 VLLM_VENV=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13 |
| neuronx-cc / NxDI / vllm-neuron / torch-neuronx | 2.23.6484.0+3b612583 / 0.8.16251+f3ca5575 / 0.4.1 / 2.9.0.2.12.22436+0f1dac25 |
| Runtime / driver | 2.30.51.0-faafe26f0 / 2.26.5.0 |
| Updated | 2026-10-29T01:03:05Z |

## Config
```
MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0   # ungated Llama-arch; for gated models export HF_TOKEN on the box
MODEL_TYPE=llama
TPS="1 2"            # tensor-parallel degrees to compare; trn1.2xlarge = 1 chip = 2 NeuronCore-v2
BATCHES="1 4"        # every TP x batch pair is a separate compile (~3 min each on trn1.2xlarge)
MAX_CTX=512          # prefill (context-encoding) length
SEQ_LEN=1024         # prompt + generated tokens
PAD_ID=2
PROMPT="Explain how a systolic array computes a matrix multiply, step by step."
VLLM_TP=2
VLLM_INPUT_LEN=256
VLLM_OUTPUT_LEN=128
VLLM_NUM_PROMPTS=64
VLLM_CONCURRENCY="1 4"
S3_BUCKET=           # optional: s3://bucket/prefix for raw profiles (NEFF/NTFF are too big for git)
```

## Steps
| Step | Exit | Duration | Note |
|---|---|---|---|
| inventory | 0 | 60s |  |
| model | 0 | 9s |  |
| bench_tp1_b1 | 0 | 295s |  |
| bench_tp1_b4 | 0 | 273s |  |
| bench_tp2_b1 | 0 | 158s |  |
| bench_tp2_b4 | 0 | 189s |  |
| profile_tp1 | 0 | 59s |  |
| profile_tp2 | 0 | 58s |  |
| vllm_tp2 | 0 | 338s |  |
| roofline | 0 | 0s |  |

## NxDI latency (p50 ms)
| TP | Batch | Prefill | Decode/step | E2E | E2E tok/s |
|---|---|---|---|---|---|
| 1 | 1 | 24.94 | 9.26 | 4959 | 207 |
| 1 | 4 | 118.34 | 9.93 | 5437 | 754 |
| 2 | 1 | 17.52 | 4.92 | 2738 | 374 |
| 2 | 4 | 60.33 | 5.26 | 2987 | 1371 |

## vLLM serving
| Result | TTFT p50 / p99 ms | TPOT p50 ms | ITL p50 ms | Output tok/s | Req/s |
|---|---|---|---|---|---|
| vllm_tp2_c1 | 24.4 / 25.0 | 7.64 | 7.64 | 128 | 1.00 |
| vllm_tp2_c4 | 55.3 / 84.7 | 7.81 | 7.55 | 490 | 3.82 |

## profile_files_tp1 (bytes, path)
```
  66571120  /home/ubuntu/profiles/20261029T003425Z/tp1/i-0422bf7dd5b82526a_pid_4800/1106284636602413_vnc_0.ntff
  34875146  /home/ubuntu/profiles/20261029T003425Z/tp1/i-0422bf7dd5b82526a_pid_4800/241226625440674_vnc_0.ntff
  60796473  /home/ubuntu/profiles/20261029T003425Z/tp1/i-0422bf7dd5b82526a_pid_4800/948584188481322_vnc_0.ntff
    963584  /home/ubuntu/profiles/20261029T003425Z/tp1/i-0422bf7dd5b82526a_pid_4800/neff_1106284636602413.neff
   3073024  /home/ubuntu/profiles/20261029T003425Z/tp1/i-0422bf7dd5b82526a_pid_4800/neff_241226625440674.neff
   2919424  /home/ubuntu/profiles/20261029T003425Z/tp1/i-0422bf7dd5b82526a_pid_4800/neff_948584188481322.neff
```

## profile_files_tp2 (bytes, path)
```
  31660469  /home/ubuntu/profiles/20261029T003425Z/tp2/i-0422bf7dd5b82526a_pid_4933/669410348316785_vnc_0.ntff
  31661064  /home/ubuntu/profiles/20261029T003425Z/tp2/i-0422bf7dd5b82526a_pid_4933/669410348316785_vnc_1.ntff
  18902049  /home/ubuntu/profiles/20261029T003425Z/tp2/i-0422bf7dd5b82526a_pid_4933/799870647384306_vnc_0.ntff
  18901091  /home/ubuntu/profiles/20261029T003425Z/tp2/i-0422bf7dd5b82526a_pid_4933/799870647384306_vnc_1.ntff
  30102319  /home/ubuntu/profiles/20261029T003425Z/tp2/i-0422bf7dd5b82526a_pid_4933/948584188481322_vnc_0.ntff
  30101907  /home/ubuntu/profiles/20261029T003425Z/tp2/i-0422bf7dd5b82526a_pid_4933/948584188481322_vnc_1.ntff
    676864  /home/ubuntu/profiles/20261029T003425Z/tp2/i-0422bf7dd5b82526a_pid_4933/neff_669410348316785.neff
   1977344  /home/ubuntu/profiles/20261029T003425Z/tp2/i-0422bf7dd5b82526a_pid_4933/neff_799870647384306.neff
   1537024  /home/ubuntu/profiles/20261029T003425Z/tp2/i-0422bf7dd5b82526a_pid_4933/neff_948584188481322.neff
```
