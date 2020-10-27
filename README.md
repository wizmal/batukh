# batukh

![license](https://img.shields.io/github/license/koshurnizam/batukh)
![gh-release](https://img.shields.io/github/v/release/koshurnizam/batukh?include_prereleases)
![pypi-ver](https://img.shields.io/pypi/v/batukh)

Detection of Languages using CRNN.

## Installation

Using [pip](http://pypi.org)

For tensorflow only installation:

`pip install batukh[tf]`

For pytorch only installation:

`pip install batukh[torch]`

For tensorflow and pytorch installation:

`pip install batukh[full]`



> :heavy_exclamation_mark: **Warning**:
>
> A simple `pip install batukh` will install neither tensorflow nor pytorch dependencies.
> 


## Training

After all the dependencies have been installed, you can train any model.


For Page Extraction(tensorflow):

```python
>>> from batukh.tensorflow.segmenter import PageExtractor
>>> page_extractor = PageExtractor()
>>> page_extractor.load_data("/path/to/data/")
>>> page_extractor.train(n_epochs=10, batch_size=1,repeat=1)
Initializing from scratch

Epoch: 1. Traininig: 100%|██████████| 70/70 [00:02<00:00, 23.95it/s, loss=0.0708]
Epoch: 2. Traininig: 100%|██████████| 70/70 [00:02<00:00, 24.35it/s, loss=0.0682]

Model saved to /content/tf_ckpts/Fri Oct 16 08:23:13 2020/ckpt-14280

Epoch: 3. Traininig: 100%|██████████| 70/70 [00:02<00:00, 23.69it/s, loss=0.0658]
Epoch: 4. Traininig: 100%|██████████| 70/70 [00:02<00:00, 24.74it/s, loss=0.0636]

Model saved to /content/tf_ckpts/Fri Oct 16 08:23:13 2020/ckpt-14420

Epoch: 5. Traininig: 100%|██████████| 70/70 [00:02<00:00, 23.68it/s, loss=0.0616]
Epoch: 6. Traininig: 100%|██████████| 70/70 [00:02<00:00, 23.95it/s, loss=0.0597]

Model saved to /content/tf_ckpts/Fri Oct 16 08:23:13 2020/ckpt-14560

Epoch: 7. Traininig: 100%|██████████| 70/70 [00:03<00:00, 23.24it/s, loss=0.0579]
Epoch: 8. Traininig: 100%|██████████| 70/70 [00:03<00:00, 23.23it/s, loss=0.0563]

Model saved to /content/tf_ckpts/Fri Oct 16 08:23:13 2020/ckpt-14700

Epoch: 9. Traininig: 100%|██████████| 70/70 [00:02<00:00, 23.44it/s, loss=0.0548]
Epoch: 10. Traininig: 100%|██████████| 70/70 [00:02<00:00, 23.54it/s, loss=0.0533]

Model saved to /content/tf_ckpts/Fri Oct 16 08:23:13 2020/ckpt-14840
```

For OCR(tensorflow):

```python
>>> from batukh.tensorflow.ocr import OCR
>>> m = OCR()
>>> m.load_data("/path/to/data")
>>> m.train(10,batch_size=1,repeat=1)       
Initializing from scratch

Epoch: 1. Traininig: 100%|██████████| 70/70 [00:04<00:00, 15.84it/s, loss=37.1]
Epoch: 2. Traininig: 100%|██████████| 70/70 [00:02<00:00, 23.57it/s, loss=29.7]

Model saved to tf_ckpts/Fri Oct 16 09:44:35 2020/ckpt-140

Epoch: 3. Traininig: 100%|██████████| 70/70 [00:02<00:00, 24.01it/s, loss=26.8]
Epoch: 4. Traininig: 100%|██████████| 70/70 [00:02<00:00, 23.84it/s, loss=25.3]

Model saved to tf_ckpts/Fri Oct 16 09:44:35 2020/ckpt-280

Epoch: 5. Traininig: 100%|██████████| 70/70 [00:02<00:00, 23.46it/s, loss=24.4]
Epoch: 6. Traininig: 100%|██████████| 70/70 [00:02<00:00, 24.33it/s, loss=23.8]

Model saved to tf_ckpts/Fri Oct 16 09:44:35 2020/ckpt-420

Epoch: 7. Traininig: 100%|██████████| 70/70 [00:02<00:00, 23.96it/s, loss=23.3]
Epoch: 8. Traininig: 100%|██████████| 70/70 [00:02<00:00, 23.67it/s, loss=22.9]

Model saved to tf_ckpts/Fri Oct 16 09:44:35 2020/ckpt-560

Epoch: 9. Traininig: 100%|██████████| 70/70 [00:03<00:00, 23.22it/s, loss=22.6]
Epoch: 10. Traininig: 100%|██████████| 70/70 [00:02<00:00, 23.52it/s, loss=22.3]

Model saved to tf_ckpts/Fri Oct 16 09:44:35 2020/ckpt-700
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
