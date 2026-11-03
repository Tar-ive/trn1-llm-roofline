# Run 20261030T175623Z-compile

| | |
|---|---|
| Instance | i-0520a98e34ea67d9f (m7i.4xlarge, us-east-1b) |
| AMI | ami-00c45d97f3070117d |
| Repo commit | 449e930 S3 compile cache, CPU compile mode, plots; 60 min trn1 budget |
| Venvs | NXDI_VENV=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13 VLLM_VENV=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13 MODE=compile BUCKET=neuron-lab-<aws-account-id>-us-east-1 |
| neuronx-cc / NxDI / vllm-neuron / torch-neuronx | 2.23.6484.0+3b612583 / 0.8.16251+f3ca5575 / 0.4.1 / 2.9.0.2.12.22436+0f1dac25 |
| Runtime / driver | 2.30.51.0-faafe26f0 / msg="Failed |
| Updated | 2026-10-30T18:19:16Z |

## Config
```
LAUNCH_MINUTES=60            # trn1.2xlarge, ~$1.34/hr
COMPILE_LAUNCH_MINUTES=120   # m7i.4xlarge, ~$0.81/hr
POWEROFF_WHEN_DONE=1
MODEL_ID=NousResearch/Meta-Llama-3.1-8B-Instruct   # ungated mirror of meta-llama/Llama-3.1-8B-Instruct
DOWNLOAD_EXCLUDE="original/*"                      # skip the duplicate 16 GB consolidated.pth checkpoint
MODEL_TYPE=llama
TPS="2"              # TP=1 can't hold 16 GB of weights on one 16 GB NeuronCore
BATCHES="1 16"       # latency floor vs. throughput; KV at b16 x 2048 tokens ~4 GB (32 layers x 8 KV heads x 128 x bf16)
MAX_CTX=1024         # prefill (context-encoding) length
SEQ_LEN=2048         # prompt + generated tokens
PAD_ID=2
PROMPT="Explain how a systolic array computes a matrix multiply, step by step."
NXDI_EXTRA="--save-sharded-checkpoint"   # shard once at compile; later loads read per-rank shards (less host RAM)
COMPILE_MIN=30       # minutes a compile step needs left before it starts (a cache hit only takes ~2-4 min)
SWAP_GB=32           # host has 30 GiB RAM; swap is insurance for loading/sharding a 16 GB checkpoint
VLLM_ENABLE=1
VLLM_TP=2
VLLM_MAX_SEQS=8
VLLM_INPUT_LEN=512
VLLM_OUTPUT_LEN=256
VLLM_NUM_PROMPTS=64
VLLM_CONCURRENCY="1 8"
VLLM_MIN=35
S3_BUCKET=           # raw profile traces; empty = s3://<lab bucket>/profiles (NEFF/NTFF are too big for git)
```

## Steps
| Step | Exit | Duration | Note |
|---|---|---|---|
| inventory | 0 | 52s |  |
| swap | 0 | 1s |  |
| model | 0 | 48s |  |
| compile_tp2_b1 | 0 | 395s | cache miss: compiling compile_seconds=364 rc=0 uploaded to s3://neuron-lab-<aws-account-id>-us-east-1/compiled/Meta-Llama-3.1-8B-Instruct/tp2-b1-ctx1024-seq2048-a5e23979ca in 29s peak_host_mem_mb=26217 (ram 26217 + swap 0)  |
| compile_tp2_b16 | 0 | 582s | cache miss: compiling compile_seconds=553 rc=0 uploaded to s3://neuron-lab-<aws-account-id>-us-east-1/compiled/Meta-Llama-3.1-8B-Instruct/tp2-b16-ctx1024-seq2048-949b2a9e8b in 28s peak_host_mem_mb=28002 (ram 28002 + swap 0)  |
