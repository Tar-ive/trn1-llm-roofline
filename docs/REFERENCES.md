# References

Papers and docs behind this project, grouped by the part of the experiment they inform.
There is no peer-reviewed paper on Llama-3.1-8B inference on Trainium1; the closest work is in section 1,
and Trainium hardware details come from AWS docs (section 7).

## 1. Trainium-specific

| Reference | Hardware | Relevance |
|---|---|---|
| NeuronMLP: SVD compression + tiling for LLM inference on Trainium, Song et al. [arXiv 2510.25977](https://arxiv.org/abs/2510.25977) | trn1.2xlarge, NxD Inference, TP=2 | **Our exact setup, Llama-3.1-8B included.** Fused low-rank MLP NKI kernels: 1.35x kernel / 1.21x end-to-end over AWS's NKI matmul (compression 0.05, lossy). Lessons: data movement beats arithmetic, avoid transposes (stationary operand must be transposed), keep intermediates in the 24 MB SBUF, block sizes past SBUF cause spills. Metrics match ours (MFU, HBM-SBUF traffic, spill bytes). |
| Axon: synthesizing superoptimizer for tensor programs, Kothari (UIUC) + AWS. [arXiv 2606.26344](https://arxiv.org/abs/2606.26344) | trn1.32xlarge, 1 NeuronCore, compiler 2.21 | Where the compiler leaves speed on the table: SiLU 1.90x, GLU 1.77x, SwiGLU 1.67x, RMSNorm 1.62x, Softmax 1.52x geomean over neuronx-cc; fused multi-op kernels up to 19x; up to 1.35x over hand-written NKI. A ready list of fusion targets. No code release. |
| NKI-Agent: fine-tuning + compile-verify-fix loop for NKI kernels, Tang et al., DL4C @ ICML 2026. [arXiv 2607.04395](https://arxiv.org/abs/2607.04395) | Trn1 | 77.3% kernel pass rate with an agent loop vs 6% single-shot: write kernels with a compile/test loop, not in one shot. NKIBench (250 tasks; 150-task subset in the main eval). Release status unclear. |
| DIFFLET: diffusion-transformer inference engine, Ma et al. [arXiv 2610.23047](https://arxiv.org/abs/2610.23047) | Trn2 | Not LLMs, but the same constraints: fixed-shape AOT compilation, fused attention needed to compile at long sequence (32.8k tokens), head-sharded attention, keeping programs resident (1.5-3.4x vs reloading). |
| HLAT: LLMs pre-trained on AWS Trainium, Fan et al. [arXiv 2404.10630](https://arxiv.org/abs/2404.10630) | trn1.32xlarge | Training, not inference; most detailed published account of Trainium1 and its stack at scale. |
| NeuronMM (UC Merced). [slides](https://docs.google.com/presentation/d/1yr1YazSJrUKHC8SZdykHWJIR5lIYYlaw/htmlpresent) | trn1.2xlarge, NxDI | Earlier presentation of the NeuronMLP line of work. Not a paper. |
| ASPLOS/EuroSys 2025 NKI Llama contest. [contest page](https://github.com/asplos-contest/2025/blob/main/OPTNKI.md), code: [aws-samples/nki-llama](https://github.com/aws-samples/nki-llama) | single trn1 chip | Llama-3.2-1B implemented with NKI kernels; NeuronMLP reports 16-26% lower latency than the top three entries. The best hands-on starting point for kernel work. |

## 2. The model

| Reference | Relevance |
|---|---|
| The Llama 3 Herd of Models. [arXiv 2407.21783](https://arxiv.org/abs/2407.21783) | Architecture and shapes behind the roofline math (`bench/roofline.py`). |
| GQA, Ainslie et al. [arXiv 2305.13245](https://arxiv.org/abs/2305.13245) | 8 KV heads for 32 query heads: why the KV cache at batch 16 x 2048 tokens is only ~4 GB. |

## 3. Hardware-limit analysis (`roofline.py`, `profile_summary_*.txt`)

| Reference | Relevance |
|---|---|
| Roofline: An Insightful Visual Performance Model, Williams, Waterman & Patterson, *CACM* 2009. [doi:10.1145/1498765.1498785](https://doi.org/10.1145/1498765.1498785) | The method: measured time vs memory-bandwidth and compute floors. |
| Efficiently Scaling Transformer Inference, Pope et al. [arXiv 2211.05102](https://arxiv.org/abs/2211.05102) | Why decode is memory-bound and prefill compute-bound; cost of tensor parallelism. Best single paper for reading our profiles. |
| LLM Inference Unveiled: Survey and Roofline Model Insights. [arXiv 2402.16363](https://arxiv.org/abs/2402.16363) | Roofline applied per inference phase and per optimization. |
| Megatron-LM, Shoeybi et al. [arXiv 1909.08053](https://arxiv.org/abs/1909.08053) | The tensor-parallel scheme NxDI uses; explains the per-step all-reduce (7-31% of a TinyLlama decode step). |
| In-Datacenter Performance Analysis of a Tensor Processing Unit, Jouppi et al., ISCA 2017. [arXiv 1704.04760](https://arxiv.org/abs/1704.04760) | Systolic-array matmul units; Trainium's TensorE is the same family. |

## 4. Optimizations in the planned sweep

| Knob (`inference_demo` flag) | Reference |
|---|---|
| `--attn-kernel-enabled`, `--attn-block-*-nki-kernel-enabled` | FlashAttention [2205.14135](https://arxiv.org/abs/2205.14135), FlashAttention-2 [2307.08691](https://arxiv.org/abs/2307.08691) |
| `--flash-decoding-enabled` | FlashDecoding++ [2311.01282](https://arxiv.org/abs/2311.01282) (original Flash-Decoding was a 2023 blog post) |
| `--quantized`, `--quantization-dtype`, `--quantization-type` | LLM.int8() [2208.07339](https://arxiv.org/abs/2208.07339), SmoothQuant [2211.10438](https://arxiv.org/abs/2211.10438), GPTQ [2210.17323](https://arxiv.org/abs/2210.17323), AWQ [2306.00978](https://arxiv.org/abs/2306.00978) |
| `--kv-cache-quant` | KIVI [2402.02750](https://arxiv.org/abs/2402.02750) |
| `--enable-eagle-speculation`, `--speculation-length` | Speculative decoding, Leviathan et al. [2211.17192](https://arxiv.org/abs/2211.17192); EAGLE [2401.15077](https://arxiv.org/abs/2401.15077) |
| `--fused-qkv`, `--qkv-kernel-enabled`, `--mlp-kernel-enabled` | Operator fusion to cut HBM round-trips; see NeuronMLP (section 1) and the NKI docs (section 7) |

## 5. Serving (`vllm_tp*` step)

| Reference | Relevance |
|---|---|
| vLLM / PagedAttention, Kwon et al., SOSP 2023. [arXiv 2309.06180](https://arxiv.org/abs/2309.06180) | The serving engine; block-based KV cache. |
| Orca, Yu et al., OSDI 2022. [paper](https://www.usenix.org/conference/osdi22/presentation/yu) | Continuous (iteration-level) batching. |
| Sarathi-Serve [2403.02310](https://arxiv.org/abs/2403.02310), Splitwise [2311.18677](https://arxiv.org/abs/2311.18677), DistServe [2401.09670](https://arxiv.org/abs/2401.09670) | Prefill/decode interference under load: the lens for TTFT and TPOT at concurrency 8. |

## 6. Where to start

1. Pope et al. and the roofline paper: what the current numbers mean.
2. NeuronMLP: kernel work on this exact hardware and model. Then Axon for which operators to fuse.
3. aws-samples/nki-llama: working NKI kernels for a Llama model on one trn1 chip.
4. The paper for whichever knob a run is testing (section 4).

## 7. AWS primary sources (no microarchitecture paper exists)

- [AWS Neuron documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/): Trainium / NeuronCore-v2 architecture pages, NKI programming guide (SBUF, PSUM, engines, DMA).
- [Neuron inference overview](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/_sources/libraries/vllm-neuron/neuron-inference-overview.rst)
- [NKI Library release notes (Neuron 2.27)](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.28.0/_sources/release-notes/prev/2.27.0/nki-lib.rst): AWS's pre-optimized kernels.
- [re:Invent 2025 AIM414: Performance engineering on Neuron with NKI](https://dev.to/kazuya_dev/aws-reinvent-2025-performance-engineering-on-neuron-how-to-optimize-your-llm-with-nki-aim414-3fni)
- [Neuron profiler / Neuron Explorer: how to profile a workload](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-explorer/how-to-profile-workload.html)
- [Build on Trainium: kernels call for proposals](https://www.amazon.science/research-awards/call-for-proposals/build-on-trainium-kernels-for-ml-acceleration-call-for-proposals-spring-2026): AWS research funding for kernel work.
