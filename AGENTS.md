# LineRemoverNN — Agent Quick Reference

A PyTorch-based system for removing ruled lines from handwritten text images using a U-Net architecture. Reconstructs overlapping characters and restores paper background.

---

## 1. Architecture Overview

### 1.1 Core Model: `LineRemovalUNet`

Located: `src/lineremovernn/model/lineremover.py`

**Structure:**

- **Encoder (Downsampling):** 4 stages → channels: 1 → 32 → 64 → 128 → 256
- **Bottleneck:** DoubleConv(256 → 512)
- **Decoder (Upsampling):** 4 stages with skip connections from encoder
- **Output:** 1-channel line mask via Conv2d + Sigmoid

**Forward pass logic:**

```python
# Encoder produces e1..e4, p1..p4
# Bottleneck produces b
# Decoder concatenates encoder features with upsampled decoder features
d4 = concat([e4, up4(b)]) → dec4
d3 = concat([e3, up3(d4)]) → dec3
...
# Final: add predicted line mask to input to restore white background
clean_output = clamp(x + line_mask, 0, 1)
```

### 1.2 Training Pipeline

Located: `src/lineremovernn/commands/train.py`

**Training hyperparameters:**

- Optimizer: `AdamW`, lr=1e-4
- Scheduler: CosineAnnealingLR (T_max = total_steps, eta_min=1e-6)
- Batch size: default 6
- Epochs: default 25 (note: `-e` flag is **1-indexed**, so `-e 25` means 25 epochs, but code does `args.epoch - 1`)
- AMP: Uses `GradScaler` with fp16 if `_probe_autocast()` returns True
- CUDA: `cudnn.benchmark = True`
- DataLoader: batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_pages

**Loss function:** `criterion(pred, clean)` = L1_loss _ 1.0 + SSIM_loss _ 0.3  
(Defined in `src/lineremovernn/utils/loss.py`)

### 1.3 Data Pipeline

**Dataset types:**

- **IAM:** Handwritten text dataset (PNG images + metadata)
- **MathWriting:** Mathematical expressions (InkML vectors → Cairo rasterization)
- **AI2D:** Diagrams

**Synthetic page generation:**

- Implemented in C++ via nanobind (`src/lineremovernn_ext/`)
- Command: `pixi run lineremovernn generate-pages`
- Uses OpenCV + Cairo for rendering
- Can generate document-like layouts with paragraphs, schemas, etc.

---

## 2. Exact Commands & Their Order

### 2.1 Setup & Build Order

```bash
# 1. Install pixi environment (includes CMake, OpenCV, Cairo)
pixi install

# 2. Install pre-commit hooks
pixi run hooks

# 3. Generate compile_commands.json for IDE support
pixi run clangd-setup

# 4. Build the Python package (compiles C++ bindings via scikit-build-core)
pixi run build

# 5. Verify installation
pixi run lineremovernn ls-models
```

### 2.2 Dataset Download & Generation

```bash
# Download IAM dataset (first time)
pixi run lineremovernn download-dataset -d iam

# Download other datasets
pixi run lineremovernn download-dataset -d mathwriting
pixi run lineremovernn download-dataset -d ai2d

# Generate synthetic training pages (default: 50 pages from IAM)
pixi run lineremovernn generate-pages

# Generate with specific datasets and options
# Format: dataset:proportion:flags where flags are 'p' (preload) and 'i' (index)
pixi run lineremovernn generate-pages --datasets iam:1:ip mathwriting:0.3 -n 100 -a -il -m -d -w 4
# Flags:
#   -a: use arc lines (not straight)
#   -il: imperfect/noisy lines
#   -m: save XML metadata
#   -d: document-like layout (paragraphs)
#   -w: CPU workers
```

### 2.3 Training

```bash
# Train for 25 epochs, batch size 6
pixi run lineremovernn train -e 25 -b 6

# Resume training from latest checkpoint
pixi run lineremovernn train -e 50 -l

# Train with extended augmentations (more aggressive transforms)
pixi run lineremovernn train -e 25 -b 6 -ex

# Custom dataset path (overrides saved/datasets/)
pixi run lineremovernn train -e 25 -b 6 -d /path/to/custom/dataset
```

### 2.4 Testing & Evaluation

```bash
# Test model on N images, show loss per image
pixi run lineremovernn test -n 10 -b 4 -l

# Preview dataset samples with optional random transforms
pixi run lineremovernn preview-dataset -n 5 -t
```

### 2.5 Model Inspection

```bash
# List available models in saved/models/
pixi run lineremovernn ls-models

# Detailed model info (layer sizes, etc.)
pixi run lineremovernn model-info
```

### 2.6 GUI Inference

```bash
# Interactive GUI for inference
pixi run lineremovernn gui-infer
```

---

## 3. Directory Tree

```
LineRemoverNN/
├── src/lineremovernn/                     # Python package
│   ├── __init__.py                        # Package init
│   ├── __main__.py                        # CLI entry point
│   ├── model/                             # Neural network architecture
│   │   └── lineremover.py                 # LineRemovalUNet
│   ├── data/                              # Dataset classes
│   │   ├── __init__.py
│   │   ├── dataset.py                     # Base TorchDataset
│   │   ├── iam.py                         # IAM dataset
│   │   ├── mathwriting.py                 # MathWriting dataset
│   │   ├── ai2d.py                        # AI2D dataset
│   │   └── pages.py                       # Generated pages + collate
│   ├── commands/                          # CLI subcommands
│   │   ├── command.py                     # Command base class
│   │   ├── download_dataset.py            # download-dataset
│   │   ├── generate_pages.py              # generate-pages
│   │   ├── train.py                       # train
│   │   ├── test.py                        # test
│   │   ├── preview_dataset.py             # preview-dataset
│   │   ├── list_models.py                 # ls-models
│   │   ├── model_info.py                  # model-info
│   │   └── gui.py                         # gui-infer
│   └── utils/                             # Utilities
│       ├── __init__.py
│       ├── consts.py                      # Runtime constants (HAS_EXT, DEVICE, etc.)
│       ├── loss.py                        # L1 + SSIM loss
│       ├── saver.py                       # Model checkpointing
│       ├── logging.py                     # Logging setup
│       └── saver.py                       # Model saving/loading
│
├── src/lineremovernn_ext/                 # C++ bindings (nanobind)
│   ├── bindings.cpp                       # Python bindings entry
│   ├── page_gen.cpp                       # Page generation logic
│   ├── page_gen.h                         # Page gen header
│   ├── datasets/                          # Dataset handling
│   │   ├── factory.cpp                    # Dataset factory
│   │   ├── iam.cpp                        # IAM C++ code
│   │   ├── mathwriting.cpp                # MathWriting C++ code
│   │   ├── ai2d.cpp                       # AI2D C++ code
│   │   └── utils.cpp                      # Shared utils
│   ├── generation/                        # Page rendering
│   │   ├── rendering.cpp                  # Image rendering
│   │   ├── asset_selection.cpp            # Crop selection
│   │   ├── layout.cpp                     # Layout construction
│   │   └── serializer.cpp                 # XML serialization
│   ├── logging/
│   │   └── python_logger.cpp              # Python logger bridge
│   └── pugixml/                           # XML parsing
│
├── pixi.toml                              # Conda environment (pixi)
├── pyproject.toml                         # Python package config
├── CMakeLists.txt                         # CMake build config
├── .pre-commit-config.yaml                # Pre-commit hooks
├── .gitignore                             # Git ignore
├── README.md                              # User documentation
└── saved/                                 # Data output directory
    ├── downloads/                         # Downloaded datasets
    ├── datasets/                          # Generated training pages
    ├── models/                            # Trained checkpoints
    └── logs/                              # Training logs
```

---

## 4. Critical Conventions & Gotchas

### 4.1 Epoch Counting Off-By-One

**In `train.py`:** `args.epoch = args.epoch - 1`  
The `-e` flag expects 1-indexed epochs, but the loop is 0-indexed. This is intentional.

### 4.2 Data Directory Structure

Generated datasets must follow:

```
saved/datasets/pages/
├── ruled-pages/    # Images with lines
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
├── clean-pages/    # Same images, lines removed
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
└── labels/         # Optional XML metadata (generated with -m)
    ├── 0.xml
    ├── 1.xml
    └── ...
```

### 4.3 Transform Consistency

When loading labels (`--load-label` or `-m` flag), the XML metadata is used. Without it, transforms are applied without ground truth.

### 4.4 GPU Autocast (AMP)

The system probes fp16 safety at startup (`_probe_autocast()` in `consts.py`).

- Runs a small synthetic test on the real model
- Disables AMP if results are non-finite or diverge >5% from fp32
- This handles GTX 16xx and other problematic GPU/cuDNN combos

### 4.5 C++ Binding Build Order

The C++ module (`_lineremovernn_ext`) is compiled by `scikit-build-core` during `pixi run build`.

- Requires nanobind, cmake, OpenCV, Cairo
- Generates stubs at `src/lineremovernn/_lineremovernn_ext.pyi`
- Must rebuild with `pixi run build` if C++ code changes

### 4.6 Environment Variables

No explicit environment variables needed. Everything is handled by:

- `pixi` (C++ deps)
- `uv` (Python deps)
- Runtime detection (device, CUDA, AMP)

### 4.7 Pre-commit Hooks

Run these in order:

1. `uv-lock` — lockfile check
2. `ruff check --fix` — linting with auto-fix
3. `ruff format` — formatting
4. `ty` — type checking

Install: `pixi run hooks`

---

## 5. File Ownership Summary

| Directory                     | Owner     | Purpose                               |
| ----------------------------- | --------- | ------------------------------------- |
| `src/lineremovernn/model/`    | Core ML   | U-Net architecture                    |
| `src/lineremovernn/commands/` | CLI       | Training, testing, inference commands |
| `src/lineremovernn/data/`     | Data      | Dataset classes and transforms        |
| `src/lineremovernn/utils/`    | Utilities | Loss, saving, logging                 |
| `src/lineremovernn_ext/`      | C++       | High-performance page generation      |
| `saved/datasets/pages/`       | Generated | Training data (ruled + clean pairs)   |
| `saved/models/`               | Generated | Model checkpoints                     |

---

## 6. External Dependencies

- **Pixi:** Conda-based dependency manager for C++ tools (OpenCV, Cairo)
- **UV:** Python package manager (replaces pip/venv)
- **PyTorch:** CUDA 13.2 wheel (specified in `pyproject.toml`)
- **Nanobind:** C++/Python bindings

---

## 7. Testing & Verification

**Run a single test:**

```bash
pixi run lineremovernn test -n 5 -l
```

**Verify model checkpoint:**

```bash
# Check saved/models/ for latest.pth
pixi run lineremovernn ls-models
```

**Preview generated data:**

```bash
pixi run lineremovernn preview-dataset -n 10 -t
```

---

## 8. Debugging Tips

- Enable debug logging in `consts.py` or via `--debug` flags
- Use `pixi run clangd-setup` for C++ IDE support
- Check `saved/logs/` for training run logs
- For C++ build issues: rebuild with `pixi run build`
- For GPU issues: check `_probe_autocast()` output in startup logs
