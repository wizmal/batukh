# KoshurRecognition

Detection of Koshur Language using CRNN.

Using Pip

`pip install ...`

After all the dependencies have been installed, you can run any model.

For Baseline Detection:

```python
>>> from KoshurRecognition.torch.segmenter import BaselineDetector
>>> m = BaselineDetector()
<All keys matched successfully>
>>> m.load_data("temp")
>>> m.train(n_epochs=10, device="cpu")
```