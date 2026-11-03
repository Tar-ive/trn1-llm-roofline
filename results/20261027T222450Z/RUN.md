# Run 20261027T222450Z

| | |
|---|---|
| Instance | i-0de00ab5229d7de4f (trn1.2xlarge, us-east-1f) |
| AMI | ami-00c45d97f3070117d (DLAMI Neuron PyTorch Inference vLLM 0.13, Ubuntu 24.04, 20260227 = Neuron 2.28) |
| Repo commit | c6bffbf run.sh: put selected venv bin/ on PATH (card generated after the fact) |
| Venvs | NXDI_VENV=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13 VLLM_VENV=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13 |
| neuronx-cc / NxDI / vllm-neuron / torch-neuronx | 2.23.6484.0+3b612583 / 0.8.16251+f3ca5575 / 0.4.1 / 2.9.0.2.12.22436+0f1dac25 |
| Runtime / driver | 2.30.51.0-faafe26f0 / 2.26.5.0 |
| Updated | 2026-10-27T23:41:05Z |

## Config
```
MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0   # ungated Llama-arch; for gated models export HF_TOKEN on the box
MODEL_TYPE=llama
TP=2                 # trn1.2xlarge = 1 Trainium chip = 2 NeuronCore-v2
BATCHES="1 4"        # each batch size is a separate compile (~5-10 min each on 8 vCPUs)
MAX_CTX=512          # prefill (context-encoding) length
SEQ_LEN=1024         # prompt + generated tokens
PAD_ID=2
PROMPT="Explain how a systolic array computes a matrix multiply, step by step."
VLLM_INPUT_LEN=256
VLLM_OUTPUT_LEN=128
VLLM_NUM_PROMPTS=64
VLLM_CONCURRENCY="1 4"
S3_BUCKET=           # optional: s3://bucket/prefix for raw profiles (NEFF/NTFF are too big for git)
```

## Steps
| Step | Exit | Duration | Note |
|---|---|---|---|
| inventory | 0 | 3s |  |
| model | 0 | 0s |  |
| bench_b1 | 0 | 175s |  |
| bench_b4 | 0 | 189s |  |
| profile_b1 | 0 | 60s |  |
| vllm_c1 | 1 | 30s | (APIServer pid=5064) pydantic_core._pydantic_core.ValidationError: 1 validation error for VllmConfig |
| vllm_c4 | 1 | 10s | (APIServer pid=5150) pydantic_core._pydantic_core.ValidationError: 1 validation error for VllmConfig |
| roofline | 0 | 0s |  |

## Latency (p50 ms)
| Batch | Prefill | Decode/step | E2E | E2E tok/s |
|---|---|---|---|---|
| 1 | 17.51 | 4.92 | 2740 | 373 |
| 4 | 60.30 | 5.26 | 2989 | 1370 |

## Profile files (bytes, path)
```
/home/ubuntu/profiles/20261027T222450Z/i-0de00ab5229d7de4f_pid_4970/669410348316785_vnc_0.ntff
/home/ubuntu/profiles/20261027T222450Z/i-0de00ab5229d7de4f_pid_4970/669410348316785_vnc_1.ntff
/home/ubuntu/profiles/20261027T222450Z/i-0de00ab5229d7de4f_pid_4970/799870647384306_vnc_0.ntff
/home/ubuntu/profiles/20261027T222450Z/i-0de00ab5229d7de4f_pid_4970/799870647384306_vnc_1.ntff
/home/ubuntu/profiles/20261027T222450Z/i-0de00ab5229d7de4f_pid_4970/948584188481322_vnc_0.ntff
/home/ubuntu/profiles/20261027T222450Z/i-0de00ab5229d7de4f_pid_4970/948584188481322_vnc_1.ntff
/home/ubuntu/profiles/20261027T222450Z/i-0de00ab5229d7de4f_pid_4970/neff_669410348316785.neff
/home/ubuntu/profiles/20261027T222450Z/i-0de00ab5229d7de4f_pid_4970/neff_799870647384306.neff
/home/ubuntu/profiles/20261027T222450Z/i-0de00ab5229d7de4f_pid_4970/neff_948584188481322.neff
```
