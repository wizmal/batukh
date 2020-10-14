# batukh

Detection of Koshur Language using CRNN.

Using Pip

`pip install ...`

After all the dependencies have been installed, you can train any model.

For Baseline Detection:

```python
>>> from batukh.torch.segmenter import BaselineDetector
>>> m = BaselineDetector()
<All keys matched successfully>
>>> m.load_data("/path/to/data")
>>> m.train(n_epochs=10, device="cpu")
```

For Word Detection:

```python
>>> from batukh.torch.ocr import WordDetector
>>> m = WordDetector()
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