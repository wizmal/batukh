_no_tf = False
_no_torch = False

try:
    import tensorflow as tf
except ImportError:
    _no_tf = True
else:
    import batukh.tensorflow.segmenter
    import batukh.tensorflow.ocr

try:
    import torch as pytorch
except ImportError:
    _no_torch = True
else:
    import batukh.torch.segmenter
    import batukh.torch.ocr

if _no_torch and _no_tf:
    raise ImportError(
        "Neither tensorflow nor pytorch installed. Please install tensorflow or pytorch or both!")
