# Batukh development handoff

This document transfers the durable project context from the Codex desktop
task to a Codex CLI session. It does not transfer the desktop task's private
conversation transcript; the checked-out branch, committed files, and this
document are the source of truth on another machine.

## Repository state

- Repository: `https://github.com/wizmal/batukh.git`
- Working branch: `testing`
- Starting revision before this work: `d0aa231`
- Scope completed: modernize the TensorFlow backend for TensorFlow 2.21 and
  Keras 3 while leaving the PyTorch implementation unchanged.
- All modernization changes and this handoff are committed on `testing`.

Before making changes, run:

```bash
git status --short
git log --oneline --decorate -5
git diff origin/master...HEAD
```

Do not reset, discard, rewrite, merge, or push changes unless the user asks.

## Non-negotiable design constraints

1. Keep the trainer custom and class-based. Training must continue to use a
   manual `tf.GradientTape` loop rather than `Model.fit()`.
2. Keep models and reusable layers as custom `keras.Model` and
   `keras.layers.Layer` subclasses.
3. Do not use `keras.Sequential`, `.add()`, or direct `.call()` invocation.
4. Keep layer creation in `__init__`; use `call()` only for the forward pass.
5. Keep TensorFlow and PyTorch as independently installable backends.
6. Do not change the PyTorch backend unless the user explicitly expands scope.
7. Target inexpensive training and inference on one NVIDIA GPU with an
   approximately 40 GB usable-memory ceiling.

## What was modernized

### TensorFlow models

- The segmentation network remains a custom ResNet-shaped encoder and
  five-stage decoder with explicit skip connections.
- The OCR network remains a custom CNN, two-layer bidirectional LSTM, and CTC
  classifier.
- Model calls propagate `training`, support Keras serialization, and work with
  Keras 3 weight files.
- OCR supports variable widths under `tf.function` while requiring a static
  input height.

### Data pipelines

- Uses `tf.data.AUTOTUNE`, current error handling, padded batches, and robust
  file validation.
- Segmentation pairs originals and labels by filename stem.
- Binary masks use a red-channel threshold; multiclass masks store zero-based
  class IDs in the red channel.
- OCR supports variable-width images, Unicode character lookup, sparse labels,
  and per-sample CTC logit lengths.
- TensorFlow-native color jitter replaces unsafe tensor mutation.

### Custom training

- Manual gradient-tape training and validation loops are retained.
- Segmentation uses batch-wide weighted categorical cross-entropy and ignores
  padded pixels.
- OCR uses sparse CTC loss with consistent blank-index handling.
- Core Keras Adam and AdamW replace TensorFlow Addons.
- Checkpointing, TensorBoard logging, prediction, Keras 3 weight loading, and
  legacy TensorFlow checkpoint loading are supported.

### Public interfaces and packaging

- TensorFlow exposes `PageExtractor`, `ImageExtractor`, `LayoutExtractor`,
  `BaselineDetector`, and `OCR`.
- Package backends are lazily imported so a TensorFlow-only installation does
  not require PyTorch.
- Packaging now provides TensorFlow and PyTorch optional extras and targets
  Python 3.10 through 3.13 for TensorFlow 2.21.
- TensorFlow Addons was removed because it is end-of-life.

## Verification already completed

The local development environment used Python 3.13 with TensorFlow 2.21.0 and
Keras 3.15.1. The following checks passed before the branch was pushed:

```text
Ruff formatting: 16 files already formatted
Ruff lint: All checks passed
Pytest: 11 passed
pip check: No broken requirements found
Package build: wheel and source distribution succeeded
PyTorch files changed: no
git diff --check: passed
```

The TensorFlow tests cover model shapes, class-based construction, dynamic OCR
widths, segmentation and OCR loaders, multiclass labels, augmentation, custom
trainer weight updates, modern and legacy checkpoints, CTC decoding, a real OCR
training step, and page-crop coordinate scaling.

Some warnings during tests originate in TensorFlow/Gast Python 3.13 AST
compatibility and Keras/NumPy copy semantics; they are not test failures.

## Local/server setup

Use Python 3.13 because TensorFlow 2.21 supports Python 3.10 through 3.13, while
Python 3.14 is not a supported target for this branch.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[tensorflow]"
python -m pip install pytest ruff build
python -m pytest -q test/tensorflow
```

For an NVIDIA server, confirm the driver and GPU before training:

```bash
nvidia-smi
python -m pip install "tensorflow[and-cuda]>=2.21,<2.22"
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

Do not begin a long training job until TensorFlow lists the GPU.

## Codex CLI setup and continuation

Install Codex CLI on the Linux server and authenticate it:

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
codex --version
codex login --device-auth
```

Run the first server-side Codex session from the repository root and paste the
prompt below. This creates a new server-side session grounded in the committed
handoff; cloning the branch does not copy the desktop task transcript.

After that first session, continue it from the same checkout with:

```bash
codex resume --last
```

Use `codex resume --all` if the session picker cannot find it after the checkout
has moved or been renamed.

## Functional understanding and recommended direction

Batukh is currently a train-your-own toolkit rather than an end-to-end document
processing system. Its components are not yet connected into a pipeline that
accepts a PDF or photo and returns structured, ordered document content.

Current measured model sizes:

- Segmentation: 33,465,794 parameters
- OCR: 10,678,897 parameters

Highest-value functional improvements, in recommended order:

1. Add task-appropriate evaluation metrics: Dice/mIoU and boundary F1 for
   segmentation, mAP for layout, baseline detection/order metrics, and CER/WER
   for OCR.
2. Package pretrained weights with class mappings, vocabulary, preprocessing,
   architecture configuration, and version metadata.
3. Add a high-level document pipeline for orientation/deskew, page extraction,
   layout analysis, baseline or line extraction, reading order, OCR, and
   structured export.
4. Replace the segmentation encoder with an efficient pretrained architecture
   such as SegFormer-B0/B1 while retaining custom Keras classes and the custom
   trainer.
5. Combine layout and embedded-image extraction using an instance-aware,
   document-specific detector inspired by DocLayout-YOLO or a compact
   RT-DETRv2 design.
6. Make baseline detection return baseline polylines, text-line polygons,
   ordered line crops, and confidence instead of only a binary probability map.
   A heatmap plus local-direction/instance head is the inexpensive first step;
   Orli-style direct ordered baseline regression is the more ambitious option.
7. Replace the CNN-BLSTM OCR model with an SVTRv2-Tiny/Small-style CTC model.
   This is the best evolutionary fit because it retains CTC and fast decoding.
8. Add PAGE XML, ALTO, JSON, and searchable-PDF export.
9. Add mixed-precision custom-loop support, compiled train steps, gradient
   clipping/accumulation, cosine warmup, early stopping, best-metric
   checkpoints, and width/resolution bucketing.

Avoid large vision-language models as the default. For the one-GPU budget,
prefer a shared 15-30M parameter layout detector, a small segmentation/baseline
model, and a 5-15M parameter CTC recognizer trained independently. Use tiled
full-page inference and FP16 mixed precision; model activations at document
resolution are a larger memory concern than parameter count.

## Suggested first prompt for Codex CLI

```text
Read HANDOFF.md completely, then inspect git status, the latest commits, and
origin/master...HEAD. Do not change anything yet. Confirm that you understand
the existing TensorFlow 2.21 modernization, the custom class-based layer/model
constraint, the custom GradientTape trainer constraint, and the one-GPU 40 GB
budget. Summarize the current TensorFlow architecture and wait for my next
implementation instruction. Do not modify the PyTorch backend, reset files,
commit, or push unless I explicitly ask.
```
