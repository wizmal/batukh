import tensorflow as tf
import time
from tqdm import tqdm
import numpy as np
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import io


class Train():
    r"""Contains methods to train a model and get predictions.

    Args:
        model           : Model to be trained.
        is_ocr (bolean) : Specifies if model is :class:`~batukh.tensorflow.utils.models.ocr_model.OCR`.
            Default : ``False``
    """

    def __init__(self, model, is_ocr=False):
        self.is_ocr = is_ocr
        self.model = model

        localtime = time.asctime()

        self.train_summary_writer = tf.summary.create_file_writer(
            f"logs/{localtime}/train")
        self.val_summary_writer = tf.summary.create_file_writer(
            f"logs/{localtime}/val")
        self.train_loss = tf.keras.metrics.Mean(name="loss", dtype=tf.float32)
        self.val_loss = tf.keras.metrics.Mean(name="loss", dtype=tf.float32)

    def _train_one_step(self, x, y):
        r"""Trains one image or batch of image.

        Args:
            x ( :class:`tensorflow.Tensor`) : Input image tensor or batch of image tensor.
            y ( :class:`tensorflow.Tensor`) : Target image tensor or batch of image tensor.

        Returns:
            :class:`tensorflow.float32` : loss.

        """
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            if self.is_ocr:
                logit_length = tf.fill(
                    [tf.shape(logits)[0]], tf.shape(logits)[1])
                loss = tf.nn.ctc_loss(labels=y, logits=logits, label_length=None,
                                      logit_length=logit_length, logits_time_major=False, blank_index=0)
                loss = tf.reduce_mean(loss)
            else:
                loss = self.criterion(y, logits)
                loss, _ = tf.nn.weighted_moments(
                    loss, (1, 2), np.sum(y*self.class_weights, axis=3))
                loss = loss[0]

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))
        return loss

    def _train(self, ds, epoch, batch_size, repeat, log_freq):
        r"""
        Args:
            ds ( :class:`tensorflow.data.Dataset`)  : dataset.
            epoch (int)      : epoch.
            batch_size (int) : Batch size of dataloader.
            repeat (int)     : Number of times dataloader will be itterated.

        """
        pbar = tqdm(total=len(ds)*repeat)
        pbar.set_description(f"Epoch: {epoch}. Traininig")

        for x, y in ds(batch_size, repeat):
            loss = self._train_one_step(x, y)
            self.train_loss.update_state(loss)
            if tf.equal(self.optimizer.iterations % log_freq, 0):
                img = tf.math.argmax(self.predict(x)[0], -1)
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.train_loss.result(),
                                      step=self.optimizer.iterations)

                    tf.summary.image("Predictions/train",
                                     self._plot_to_image(
                                         self._plot(img, y[0][:, :, 1])),
                                     step=self.optimizer.iterations)

            pbar.update(1)
            pbar.set_postfix(loss=float(self.train_loss.result()))
        pbar.close()

    def train(self, n_epochs, train_dl=None, val_dl=None,  batch_size=1, repeat=1, criterion=None, class_weights=None, optimizer=None, weight_decay=None, learning_rate=0.0001, lr_decay=None, save_checkpoints=True, checkpoint_freq=None, checkpoint_path=None, max_to_keep=5, log_freq=100):
        r"""
        Args:
            n_epochs (int) : Number of epochs.
            train_dl ( :class:`tensorflow.data.Dataset`,optional) : Dataset for training.
                Default: ``None``
            val_dl ( :class:`tensorflow.data.Dataset`,optional): Dataset for validation.
                Default: ``None``
            batch_size (int,optional) : batch size.
                Default: :math:`1`
            repeat (int,optional) : specifies the number of times dataset will be itterated in one epoch.
                Default : :math:`1`
            criterion (optional): Specifies loss function to be used in training.
                Default : if is_ocr is ``True`` :class:`tensorflow.nn.ctc_loss` used
                else :class:`tensorflow.nn.softmax_cross_entropy_with_logits` used.
            class_weights (list,,optional) : weights given to class while calculating loss.
                Default : Same for each class.
            optimizer (optional) : optimizer used in training.
                Default : :class:`tensorflow.keras.optimizers.Adam`
            learning_rate (float,optional):learning rate of optimizer.
                Default: :math:`0.0001`
            save_checkpoints (bol,optional) : Specifes whether to save checkpoints.
                Default: ``True``
            checkpoint_freq (int,optional) : Specifies the number of epochs after checkpoints will be saved.
                Default:  :math:`=\lfloor \frac{ \text{n_epochs}}{10}+1 \rfloor`
            checkpoint_path (str,optional): Path where  checkpoints will be saved or loaded from.
                Default :  `"tf_ckpts/{}".format(` :class:`time.asctime` `)`
            max_to_keep (int,optional) : Maximun number of latest checkpoints to keep.
                Default : :math:`5`
                """
        if lr_decay is not None:
            learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                learning_rate, decay_steps=100000, decay_rate=lr_decay, staircase=True)

        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        if weight_decay is not None:
            optimizer = tfa.optimizers.AdamW(
                learning_rate=learning_rate, weight_decay=weight_decay)
        self.optimizer = optimizer
        if criterion is None:
            if self.is_ocr:
                criterion = tf.nn.ctc_loss
            else:
                criterion = tf.nn.softmax_cross_entropy_with_logits
        self.criterion = criterion
        if class_weights is None:
            class_weights = [1]*self.model.n_classes
        self.class_weights = class_weights
        localtime = time.asctime()

        if checkpoint_path is None:
            checkpoint_path = "tf_ckpts/{}".format(localtime)
        self.checkpoint = tf.train.Checkpoint(
            model=self.model, optimizer=self.optimizer)

        if checkpoint_freq is None:
            checkpoint_freq = n_epochs//10 + 1

        self.manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=checkpoint_path,
            max_to_keep=max_to_keep)
        self.checkpoint.restore(self.manager.latest_checkpoint)

        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch")

        for epoch in range(1, n_epochs + 1):
            self._train(train_dl, epoch, batch_size, repeat, log_freq)
            if epoch % checkpoint_freq == 0:
                checkpoint_path = self.manager.save(self.optimizer.iterations)
                print("Model saved to {}".format(checkpoint_path))
            if val_dl is not None:
                with self.val_summary_writer.as_default():
                    self._val(val_dl,  batch_size, repeat, epoch, log_freq)
            self.val_loss.reset_states()
            self.train_loss.reset_states()

    def _val_one_step(self, x, y):
        r"""

        Args:
            x ( :class:`tensorflow.Tensor`) : input image tensor.
            y ( :class:`tensorflow.Tensor`) : input label tensor.

        Returns:
            :class:`tensorflow.float` :loss
            :class:`tensorflow.Tensor` : logits
            """
        logits = self.model(x, training=False)
        if self.is_ocr:
            logit_length = tf.fill(
                [tf.shape(logits)[0]], tf.shape(logits)[1])
            loss = tf.nn.ctc_loss(labels=y, logits=logits, label_length=None,
                                  logit_length=logit_length, logits_time_major=False, blank_index=0)
            loss = tf.reduce_mean(loss)
        else:
            loss = self.criterion(y, logits)
            loss, _ = tf.nn.weighted_moments(
                loss, (1, 2), np.sum(y*self.class_weights, axis=3))
            loss = loss[0]

        return loss, logits

    def _val(self, ds, epoch,  batch_size, repeat, log_freq):
        r"""
        Args:
            ds ( :class:`tensorflow.data.Dataset`) : datsset.
            epoch (int) : epoch.
            batch_size (int) : batch size.
            repeat (int): number of times dataset will be itterated in one epoch.
            """
        pbar = tqdm(total=len(ds)*repeat)
        pbar.set_description(f"Epoch: {epoch}. validation")
        for i, (x, y) in enumerate(ds(batch_size, repeat)):
            loss, _ = self._val_one_step(x, y)
            self.val_loss.update_state(loss)
            if tf.equal(((epoch-1)*len(ds)+i) % log_freq, 0):
                img = tf.math.argmax(self.predict(x)[0], -1)
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.val_loss.result(),
                                      step=self.optimizer.iterations)
                    tf.summary.image("Predictions/val",
                                     self._plot_to_image(
                                         self._plot(img, y[0][:, :, 1])),
                                     step=self.optimizer.iterations)
            pbar.update(1)
            pbar.set_postfix(loss=float(self.val_loss.result()))
        pbar.close()

    def predict(self, x):
        r"""
        Args:
            x ( :class:`tensorflow.Tensor`) : Input image tensor.

        Returns:
            :class:`tensorflow.Tensor` : predictions.
            """

        logits = self.model(x, training=False)
        return logits

    def save_model(self, path, name):
        r"""Saves model

        Args:
            path (str)          : path of the folder where model is to be saved.
            """

        self.model.save_weights(path)
        print("Model saved at "+path)

    def load_model(self, path):
        r"""Loads presaved weights.

        Args:
            path (str) : path of the model to be loaded.

        """
        self.model.load_weights(path)
        print("Model Loaded")

    def _plot_to_image(self, figure):
        """
        Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call.
        """
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return image

    def _plot(self, x, y):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(x, "gray")
        ax1.set_title("Prediction")
        ax2.imshow(y, "gray")
        ax2.set_title("Ground Truth")
        return fig
