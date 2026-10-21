# AutoDP: Automating Model Deployment on Heterogeneous Embedded Devices

> Submitted for review to IEEE TMC.

AutoDP is an end-to-end system that automates AI model deployment on heterogeneous embedded devices by resolving the _model-engine mismatch_ problem. Instead of relying on a single inference engine, AutoDP partitions a model into segments and leverages multiple inference engines to jointly optimize and execute them.

## Problem

When deploying AI models on embedded devices, not all operators are fully supported by a single inference engine. Our study across 150 open-source models shows that **25.3% fail to execute** due to model-engine mismatch:

- **Unsupported operators** missing in some engines
- **Developer-customized operators** incompatible with all engines
- **Version heterogeneity** across devices further exacerbating mismatches

## Approach

AutoDP addresses these challenges through three novel technical modules:

**1. Agent-Based Operator Analyzer** — Uses AI agents to retrieve engine documentation and source code, then checks operator-engine compatibility via LLM-based code analysis. Handles diverse naming conventions and generates custom operator registration when needed.

**2. Structure-Aware Pattern Detector** — Mines optimizable operator patterns (e.g., Conv+Add fusion) from each engine's internals. Reorganizes model copies per engine to preserve these patterns as constraints during partitioning.

**3. Greedy Cross-Engine Inference Algorithm** — Partitions the model by minimizing a marginal cost that balances operator execution latency, pattern saving credit, and engine-switching penalty. Includes bounded backtracking to escape local optima, achieving near-linear time complexity.

## Key Results

- **Deployment success rate**: Near 100% (up from <80% with single engines)
- **Deployment time reduction**: Up to **79.3%** compared with expert human developers
- **Inference latency reduction**: Over **26.6%** on average compared with SOTA engines
- Comparable task accuracy and resource consumption

## Hardware

| Type        | Device              | RAM  | VRAM | Storage |
| ----------- | ------------------- | ---- | ---- | ------- |
| Embedded    | Raspberry Pi 5      | 8GB  | -    | 32GB    |
| Embedded    | Jetson Nano         | 4GB  | 4GB  | 64GB    |
| Embedded    | IBOX-606 mini PC    | 32GB | 12GB | 256GB   |
| Edge Server | Supermicro X11SCA-F | 32GB | 16GB | 1TB     |

## Evaluated Applications

| Application                   | Model            | Device           | Dataset  |
| ----------------------------- | ---------------- | ---------------- | -------- |
| Image Recognition             | ResNet-152       | Raspberry Pi     | ImageNet |
| Speech Sentiment Analysis     | Whisper          | Jetson Nano      | TESS     |
| Vision-Language Understanding | DeepSeek-VL-1.3B | IBOX-606 mini PC | MMMU     |

## Supported Inference Engines

ONNX Runtime, TensorRT, OpenVINO, TVM, MLIR

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Configure API keys and target device address in `config.py`
3. Start the server on the target device: `python HTTP_server.py`
4. Run the pipeline: `jupyter notebook main_image_recognition.ipynb`
