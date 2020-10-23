from .utils.train import Train
from .utils.data.dataloader import OCRDataLoader
from .utils.data.augmentation import MultipleColorJitter
from .utils.models.ocr_model import OCRModel
import tensorflow as tf

# todo batch size segmenter.
# todo dynamic  weights.
# todo checkpoints.
# lr sheduler.
# l2 regulirization e-6.

# todo tensorboard
# todo example boun
#


class OCR(Train):
    r"""This class is used for octical character recognition.

    Example

    .. code:: python

        >>> from batukh.tensorflow.ocr import OCR
        >>> m = OCR(177)
        >>> m.load_data(train_path="/data/",height=32)
        >>> m.train(1)


    Args:
        n_classes (int) : number of outputs nodes of ocr model.
    """

    def __init__(self, n_classes):

        super().__init__(model=OCRModel(n_classes), is_ocr=True)
        self.train_dl = None
        self.val_dl = None

    def load_data(self, train_path, val_path=None, height=64):
        r"""Loads Train and Validation datset.

        Args:
            train_path (str)        : path of the folder contaings images folder,labels.txt and table.txt for train dataset.
            val_path (str,optional) : path of the folder contaings images folder ,labels.txt and table.txt  for validation dataset.
            """
        self.train_dl = OCRDataLoader(
            train_path, height)
        if val_path is not None:
            self.val_dl = OCRDataLoader(
                val_path, height)

    def train(self, n_epochs, train_dl=None, val_dl=None, repeat=1, criterion=None, class_weights=None, optimizer=None, learning_rate=0.0001, save_checkpoints=True, checkpoint_freq=None, checkpoint_path=None, max_to_keep=5):
        if train_dl is None:
            train_dl = self.train_dl
        if val_dl is None:
            val_dl = self.val_dl
        super().train(n_epochs, train_dl=train_dl, val_dl=val_dl, batch_size=1, repeat=repeat, criterion=criterion, class_weights=class_weights,
                      optimizer=optimizer, learning_rate=learning_rate, save_checkpoints=save_checkpoints, checkpoint_freq=checkpoint_freq, checkpoint_path=checkpoint_path, max_to_keep=max_to_keep)

    def map2string(self, inputs, table=None, blank_index=None):
        # todo : shift theese methods to ocr
        """Maps tensor to stings as per :class:`~self.inv_table`.

        Args:
            inputs (:class:`tensorflow.SparseTensor`) : Input tensors.
            table ( :class:`tensorflow.lookup.StaticHashTable`,optional) : Table according to which maping is done.
                Default: ``self.train_dl.inv_table``
            blank_index (int,optional) : Blank index.
                Default : ``self.train_dl.blank_index``



        Returns:
            list : list of strings."""
        if table is None:
            table = self.train_dl.inv_table
        if blank_index is None:
            blank_index = self.train_dl.blank_index
        inputs = tf.sparse.to_dense(inputs,
                                    default_value=blank_index).numpy()
        strings = []
        for i in inputs:
            text = [table[char_index] for char_index in i
                    if char_index != blank_index]
            strings.append(''.join(text))
        return strings

    def decode(self, inputs, from_pred=True, method='gready', merge_repeated=True, table=None, blank_index=None):
        """Decodes the model logits using ctc decoder.

        Example

        .. code:: python

            >>> from batukh.tensorflow.ocr import OCR
            >>> import tensorflow as tf
            >>> m = OCR(177)
            >>> m.load_model("/saved_model/")
            >>> x = tf.io.read_file("/image.png")
            >>> x = tf.io.decode_png(x,channels=1)
            >>> y = m.predict(x)
            >>> pred = m.decode(y)



        Args:
            inputs ( :class:`tensorflow.Tensor`) : Input tensor.
            from_pred (bol,optional): ``True`` if input is return of ( :class:`~batukh.tensorflow.ocr.OCR.predict`. ``False`` if input is a :class:`tensorflow.SparseTensor`.
                Default: ``True``
            method (str,optional)   :if  ``'gready'``  :class:`tensorflow.nn.ctc_greedy_decoder` used for decoding.if  ``'beam_search'``  :class:`tensorflow.nn.ctc_beam_search_decoder` used.
                Default: `` 'greedy' ``
            merge_repeated (bol,optional): Specifes if similar charsters will be merged.
                Default: ``True``
            table ( :class:`tensorflow.lookup.StaticHashTable`,optional) : Table according to which maping is done.
                Default: ``self.train_dl.inv_table``
            blank_index (int,optional) : Blank index.
                Default : ``self.train_dl.blank_index``

        Returns:
            list: decoded list of strings  """
        self.merge_repeated = merge_repeated
        if from_pred:
            logit_length = tf.fill([tf.shape(inputs)[0]], tf.shape(inputs)[1])
            if method == 'greedy':
                decoded, _ = tf.nn.ctc_greedy_decoder(
                    inputs=tf.transpose(inputs, perm=[1, 0, 2]),
                    sequence_length=logit_length,
                    merge_repeated=self.merge_repeated)
            elif method == 'beam_search':
                decoded, _ = tf.nn.ctc_beam_search_decoder(
                    inputs=tf.transpose(inputs, perm=[1, 0, 2]),
                    sequence_length=logit_length)
            inputs = decoded[0]

        decoded = self.map2string(decoded)
        return decoded
