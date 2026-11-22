# GPT-OSS 20B on NVIDIA Quadro K5200 — ~16 Tokens/sec

**Running GPT-OSS 20B MXFP4 on a 2014-era NVIDIA Quadro K5200 (8 GB VRAM) using a custom CUDA Compute Capability 3.5 build of llama.cpp.**

This repository documents a real-world local AI experiment demonstrating that a Kepler-generation workstation GPU can run a modern 20-billion-parameter Mixture-of-Experts (MoE) language model at usable inference speeds.

**Key result: Approximately 15–16 tokens/sec with up to 16,384 tokens of context.**

---

## Hardware Configuration

| Component | Specification |
|---|---|
| Workstation | HP Z240 Workstation |
| CPU | Intel Xeon E3-1240 v5 (4 cores / 8 threads) |
| GPU | NVIDIA Quadro K5200 |
| GPU Architecture | Kepler — Compute Capability 3.5 |
| GPU Memory | 8 GB GDDR5 |
| System Memory | 16 GB DDR4 |
| Operating System | Ubuntu Server |
| Inference Engine | llama.cpp — Custom CUDA Build |
| llama.cpp Build | b1-b367989 |
| Model Format | MXFP4 GGUF |

## AI Model

**GPT-OSS-20B-Uncensored-HauhauCS-Aggressive**

[Hugging Face — Model Repository](https://huggingface.co/HauhauCS/GPT-OSS-20B-Uncensored-HauhauCS-Aggressive)

GPT-OSS 20B uses a Mixture-of-Experts (MoE) architecture. Although the model contains approximately 20 billion total parameters, only a subset is activated per token.

---

## Real-World Benchmark Results

Two separate tests were conducted using the llama.cpp Web UI.

| Test | Context Window | Generation Speed |
|---|---:|---:|
| Self-introduction and capabilities explanation | 8,192 tokens | **16.10 tokens/sec** |
| Exa-assisted Melbourne weather research | 16,384 tokens | **15.47 tokens/sec** |

**Observed generation throughput: approximately 15–16 tokens/sec.**

These figures represent real-world, single-run generation measurements rather than standardized benchmark averages. They do not include all model-loading, prompt-processing or external tool latency.

### Test 1 — Self-Introduction and Capabilities

The model was asked to introduce itself and explain its capabilities in Turkish.

The generated response covered topics including programming, data analysis, language translation, research, finance, travel, and general assistance.

This was a text-generation test without requiring external web search.

**Measured results:**

- Context configuration: 8,192 tokens
- Generated output: 574 tokens
- Generation throughput: **16.10 tokens/sec**
- Generation time: approximately 35 seconds

### Benchmark Screenshot

![GPT-OSS 20B running at 16.10 tokens/sec on NVIDIA Quadro K5200](./benchmark-screenshot.JPG)

The screenshot captures the GPT-OSS 20B self-introduction response in the llama.cpp Web UI, alongside the `btop` system monitor on the HP Z240 workstation.

### Test 2 — Web-Assisted Weather Research

In a separate test, GPT-OSS 20B was asked to research Melbourne's seven-day weather forecast using Exa MCP.

The model produced a table containing daily temperatures, rainfall probability, rainfall amounts and wind information.

**Measured results:**

- Context configuration: 16,384 tokens
- Generation throughput: **15.47 tokens/sec**
- Web search: Exa MCP invoked successfully

The response contained some translation and interpretation mistakes. Successful tool usage does not guarantee factual accuracy, and the reported forecast was not independently verified as part of this benchmark.

---

## Model Configuration

| Parameter | Value |
|---|---|
| Context Size | 16,384 tokens for second test |
| Auto-fit | Enabled |
| GPU Offloading | 25/25 layers |
| Parallel Requests | 1 |
| Reasoning | Disabled via server configuration |
| Model Management | llama.cpp Router Mode |
| Web Integration | Exa MCP |

### Model-Specific INI Preset

File: `gpt-oss-presets.ini`

```ini
version = 1

[gpt-oss-20b]
model = /home/dr/models/gpt-oss-20b/GPT-OSS-20B-Uncensored-HauhauCS-MXFP4-Aggressive.gguf
ctx-size = 16384
parallel = 1
fit = on
fit-ctx = 16384
reasoning = off
```

### Server Launch Command

```bash
~/llamacpp-kepler/build/bin/llama-server \
  --models-dir /home/dr/models \
  --models-preset /home/dr/gpt-oss-presets.ini \
  --models-max 1 \
  --fit on \
  --reasoning off \
  --webui-mcp-proxy \
  --host 0.0.0.0 \
  --port 8081
```

The server runs in Router Mode, with a maximum of one model loaded at a time.

The paths shown reflect the original test environment and should be adapted for other installations.

---

## GPU Offloading and Memory Usage

The following measurements were captured from the llama.cpp logs while GPT-OSS 20B was configured for a 16,384-token context window.

### Actual llama.cpp Output

```text
load_tensors: offloading output layer to GPU
load_tensors: offloading 23 repeating layers to GPU
load_tensors: offloaded 25/25 layers to GPU

load_tensors: CPU_Mapped model buffer size = 10949.33 MiB
load_tensors: CUDA0 model buffer size = 6095.35 MiB

llama_kv_cache: CUDA0 KV buffer size = 384.00 MiB
llama_kv_cache: CUDA0 KV buffer size = 18.00 MiB

sched_reserve: CUDA0 compute buffer size = 398.38 MiB
```

### Memory Allocation Summary

| Memory Component | Reported Allocation |
|---|---:|
| CUDA Model Buffer | 6095.35 MiB |
| CUDA KV Cache | 402.00 MiB |
| CUDA Compute Buffer | 398.38 MiB |
| CPU-Mapped Model Buffer | 10949.33 MiB |
| GPU Layers Offloaded | 25/25 |

### Technical Interpretation

All 25 offloadable layers were assigned to the GPU.

The reported CUDA model buffer, KV cache and compute buffer total approximately **6.73 GiB**. Actual GPU memory consumption can be higher because driver and runtime allocations are not necessarily included.

The CPU-mapped model buffer was approximately **10.69 GiB**. This is a file-backed mapping measurement, not necessarily additional exclusive physical RAM usage.

CPU-mapped and GPU allocation figures must not simply be added together to determine total memory consumption.

---

## CUDA and Kepler Compatibility

The NVIDIA Quadro K5200 is based on the Kepler architecture and supports CUDA Compute Capability 3.5.

Modern CUDA software stacks may not support this older GPU architecture without modifications or compatible legacy tooling.

This experiment used a custom llama.cpp build supporting the Quadro K5200.

### Build Information

```text
llama.cpp build: b1-b367989
GPU: NVIDIA Quadro K5200
Compute Capability: 3.5
Backend: CUDA
Operating System: Ubuntu Server
```

The exact CUDA toolkit version, compiler options and CMake commands have not yet been verified for publication.

A reproducible build guide may be added after those details are recovered from the original installation.

---

## Limitations and Reproducibility

- Results were measured on a specific HP Z240 hardware configuration.
- The Quadro K5200 requires a compatible Kepler-capable CUDA build.
- The reported generation rates are observations from two workloads, not standardized benchmark averages.
- Throughput varies with prompt length, context size, quantization and inference settings.
- Model-loading time and external MCP calls add latency beyond token generation.
- The self-introduction test does not independently verify the model's claimed capabilities.
- The weather research response was not independently verified for forecast accuracy.
- Controlled multi-run benchmarking has not yet been performed.
- Successful GPU offloading does not mean all model data resides exclusively in VRAM.

---

## Conclusion

This experiment demonstrates that **GPT-OSS 20B MXFP4 can run on an HP Z240 workstation equipped with an NVIDIA Quadro K5200 8 GB GPU**, using a compatible custom llama.cpp CUDA build.

The system successfully demonstrated:

- Full offloading of 25/25 supported layers.
- A working 16,384-token context configuration.
- **16.10 tokens/sec** during a self-introduction task at 8K context.
- **15.47 tokens/sec** during an Exa-assisted research task at 16K context.
- Functional MCP integration through llama.cpp.
- Usable local inference on legacy workstation hardware.

**The main finding is that older Kepler-generation GPUs can still deliver meaningful local AI inference performance with suitable software, model architectures and memory-management techniques.**

---

### Keywords

GPT-OSS 20B, GPT-OSS MXFP4, NVIDIA Quadro K5200, NVIDIA Kepler, CUDA 3.5, llama.cpp, Local LLM, HP Z240, GPU Offloading, Mixture of Experts, Legacy GPU, AI Benchmark, Local AI Inference.
