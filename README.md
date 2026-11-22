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

The model uses a Mixture-of-Experts architecture. Although it contains approximately 20 billion total parameters, only a subset of the model's parameters is activated for each token.

This architecture can offer computational advantages compared with similarly sized dense models.

---

## Real-World Benchmark Results

| Test | Context Window | Generation Speed |
|---|---:|---:|
| Initial text inference | 8,192 tokens | **16.10 tokens/sec** |
| Web-assisted weather research | 16,384 tokens | **15.47 tokens/sec** |

**Observed generation throughput: approximately 15–16 tokens/sec.**

These results were obtained through the llama.cpp Web UI on the actual workstation.

They represent observed single-run inference measurements, not standardized multi-run benchmark averages.

Generation speed does not include all prompt-processing, model-loading, or external web-tool latency.

### Benchmark Screenshot

![GPT-OSS 20B running at 16.10 tokens/sec on NVIDIA Quadro K5200](./benchmark-screenshot.JPG)

**Screenshot details:**

- Model: GPT-OSS 20B MXFP4
- Context window: 8,192 tokens
- Generated output: 574 tokens
- Generation throughput: **16.10 tokens/sec**
- Generation time: Approximately 35 seconds
- System monitoring: btop running on Ubuntu Server

A separate test with a 16,384-token context window achieved **15.47 tokens/sec**.

---

## Model Configuration

The llama.cpp server was configured with the following settings:

| Parameter | Value |
|---|---|
| Context Size | 16,384 tokens |
| Auto-fit | Enabled |
| GPU Offloading | 25/25 layers |
| Parallel Requests | 1 |
| Reasoning | Disabled via server configuration |
| Model Management | llama.cpp Router Mode |
| Web Integration | Exa MCP |
| Web Interface | llama.cpp Web UI |

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

### llama.cpp Server Launch Command

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

The server operates in Router Mode, allowing multiple GGUF models to be discovered and selected dynamically.

The `--models-max 1` parameter limits the server to one loaded model at a time.

Auto-fit is enabled to adjust model configuration according to available device memory.

**Note:** The paths above reflect the original test environment and should be adapted for other installations.

---

## GPU Offloading and Memory Usage

The following values were captured directly from the llama.cpp model-loading logs with a 16,384-token context window.

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

The reported CUDA model buffer, KV cache, and compute buffer total approximately **6.73 GiB**.

Actual GPU memory consumption may be higher because the reported figures do not necessarily include all driver and runtime allocations.

The CPU-mapped model buffer was approximately **10.69 GiB**.

This value should not be interpreted as additional exclusive physical RAM consumption. Memory-mapped model data can be backed by file-system cache and shared memory mappings.

Similarly, CPU-mapped and CUDA model buffer values should not simply be added together to calculate total physical memory consumption.

The experiment demonstrates that successful GPU offloading does not require the model's entire file size to fit exclusively within VRAM.

---

## Web Search and Tool-Calling Test

GPT-OSS 20B was tested using **Exa MCP**, integrated through the llama.cpp Web UI.

### Example Task

Research the seven-day weather forecast for Melbourne, Australia, including temperatures, precipitation probability, rainfall amounts, wind conditions, and source information.

### Observed Results

- Successfully invoked the web search integration.
- Generated a structured seven-day weather forecast.
- Included daily temperature, precipitation and wind information.
- Produced a source URL.
- Completed the response at a reported generation speed of **15.47 tokens/sec**.

The output contained some interpretation and translation errors.

Therefore, successful tool execution should not be interpreted as proof that all generated information was factually accurate.

This experiment primarily demonstrates functional web-tool integration and usable inference throughput on legacy hardware.

---

## CUDA and Kepler Compatibility

The NVIDIA Quadro K5200 uses the Kepler architecture with CUDA Compute Capability 3.5.

Modern CUDA software stacks do not necessarily provide native support for this older architecture.

This experiment used a **custom llama.cpp build compatible with the Quadro K5200**.

### Build Information

```text
llama.cpp build: b1-b367989
GPU: NVIDIA Quadro K5200
Compute Capability: 3.5
Backend: CUDA
Operating System: Ubuntu Server
```

The exact CUDA toolkit version, compiler configuration, and CMake build commands have not yet been documented in this repository.

A reproducible build guide will be added after these details are verified from the original test environment.

No unverified CUDA compilation commands are provided here.

---

## Limitations and Reproducibility

- Results were measured on a specific HP Z240 hardware configuration.
- The GPU requires a compatible Kepler-capable CUDA build.
- Measured throughput may vary with prompt length, context size, sampling configuration, and workload.
- Model-loading time is separate from text-generation speed.
- External MCP tools can significantly increase total response latency.
- The 15–16 tokens/sec figures are observed results, not guaranteed performance.
- Controlled multi-run benchmarks have not yet been performed.
- Successful loading does not imply that every operation executes exclusively in GPU memory.

Additional benchmarking and build documentation may be added as testing continues.

---

## Conclusion

This experiment demonstrates that **GPT-OSS 20B MXFP4 can run on an HP Z240 workstation equipped with an NVIDIA Quadro K5200 8 GB GPU**, using a compatible custom llama.cpp CUDA build.

The system successfully demonstrated:

- GPU offloading of 25/25 layers.
- A 16,384-token context configuration.
- Approximately **15–16 tokens/sec** generation throughput.
- Functional Exa MCP web-tool integration.
- Usable local inference on legacy workstation hardware.

**The key finding is that older Kepler-generation GPUs can still provide meaningful local AI inference capability when paired with suitable software, model architectures, and memory-management techniques.**

---

### Keywords

GPT-OSS 20B, GPT-OSS MXFP4, NVIDIA Quadro K5200, NVIDIA Kepler, CUDA 3.5, llama.cpp, Local LLM, HP Z240, GPU Offloading, Mixture of Experts, Legacy GPU, AI Benchmark, Local AI Inference.
