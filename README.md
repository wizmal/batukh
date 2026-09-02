# batukh

Custom TensorFlow and PyTorch models for document segmentation and OCR.

## TensorFlow

The TensorFlow backend targets TensorFlow 2.21 and Python 3.10–3.13. For local
development, install it with:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[tensorflow]"
```

Segmentation datasets contain matching files under `originals/` and `labels/`.
Binary masks use a black background and a red foreground. Multiclass masks use
zero-based class IDs in the red channel.

Page extraction:

```python
>>> from batukh.tensorflow.segmenter import PageExtractor
>>> page_extractor = PageExtractor()
>>> page_extractor.load_data("/path/to/data")
>>> history = page_extractor.train(
...     n_epochs=10,
...     batch_size=2,
...     class_weights=[1.0, 100.0],
... )
```

OCR datasets contain `images/`, `labels.txt`, and `table.txt`. Each label line
uses `<image-stem>: <text>`. Each `table.txt` line defines one Unicode symbol;
the final line is reserved for the CTC blank class.

OCR:

```python
>>> from batukh.tensorflow.ocr import OCR
>>> ocr = OCR(n_classes=177)
>>> ocr.load_data("/path/to/ocr-data", height=64)
>>> history = ocr.train(n_epochs=10, batch_size=2)
>>> logits = ocr.predict(images)
>>> text = ocr.decode(logits)
```







For Baseline Detection(pytorch):

```python
>>> from batukh.torch.segmenter import BaselineDetector
>>> m = BaselineDetector()
<All keys matched successfully>
>>> m.load_data("/path/to/data")
>>> m.train(n_epochs=10, device="cpu")
```

For OCR(pytorch):

```python
>>> from batukh.torch.ocr import OCR
>>> m = OCR()
>>> m.load_data("/path/to/train_dir", "/path/to/train_labels", 
"/path/to/val_dir", "/path/to/val_labels")
Building Dictionary. . .
Building Dictionary. . .
>>> m.train(5)       
Epoch: 0. Traininig: 100%|███████████████| 140/140 [00:04<00:00, 30.18it/s, loss=2.59]
Epoch: 0. Validating: 100%|███████████████| 140/140 [00:01<00:00, 112.06it/s, loss=2.59]
Models Saved!
Epoch: 1. Traininig: 100%|███████████████| 140/140 [00:04<00:00, 32.39it/s, loss=2.36]
Epoch: 1. Validating: 100%|███████████████| 140/140 [00:01<00:00, 121.36it/s, loss=2.18]
Models Saved!
Epoch: 2. Traininig: 100%|███████████████| 140/140 [00:04<00:00, 31.12it/s, loss=2.54]
Epoch: 2. Validating: 100%|███████████████| 140/140 [00:01<00:00, 108.65it/s, loss=2.48]
Models Saved!
Epoch: 3. Traininig: 100%|███████████████| 140/140 [00:04<00:00, 31.10it/s, loss=2.48]
Epoch: 3. Validating: 100%|███████████████| 140/140 [00:01<00:00, 109.46it/s, loss=2.42]
Models Saved!
Epoch: 4. Traininig: 100%|███████████████| 140/140 [00:04<00:00, 30.17it/s, loss=2.49]
Epoch: 4. Validating: 100%|███████████████| 140/140 [00:01<00:00, 110.09it/s, loss=2.42]
Models Saved!
```
