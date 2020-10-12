import tensorflow as tf
import time
from tqdm import tqdm
import numpy as np


class Train():
    r"""Contains methods to train a model and get predictions.

    Args:
        model           : Model to be trained.
        is_ocr (bolean) : Specifies if model is OCR.Default value False.
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
            x (tf.Tensor) : Input image tensor or batch of image tensor.
            y (tf.Tensor) : Target image tensor or batch of image tensor.

        Returns:
            loss (tf.int32) : loss of logits of x w.r.t target y.

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

    def _train(self, ds, epoch, batch_size, repeat):
        r"""
        Args:
            ds (Dataloader)  : Dataloader object.
            epoch (int)      : Number of epochs in training loop.
            batch_size (int) : Batch size of dataloader.
            repeat (int)     : Number of times dataloader will be itterated.

        """
        pbar = tqdm(total=len(ds))
        pbar.set_description(f"Epoch: {epoch}. Traininig")

        for x, y in ds(batch_size, repeat):
            loss = self._train_one_step(x, y)
            self.train_loss.update_state(loss)
            pbar.update(1)
            pbar.set_postfix(loss=float(self.train_loss.result()))
        pbar.close()

    def train(self, n_epochs, train_dl=None, val_dl=None,  batch_size=2, repeat=1, criterion=None, class_weights=None, optimizer=None, learning_rate=0.0001, save_checkpoints=True, checkpoint_freq=None, checkpoint_path=None, max_to_keep=5):
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
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

            with self.train_summary_writer.as_default():
                self._train(train_dl, epoch, batch_size, repeat)
            if epoch % checkpoint_freq == 0:
                checkpoint_path = self.manager.save(self.optimizer.iterations)
                print("Model saved to {}".format(checkpoint_path))
            if val_dl is not None:
                with self.val_summary_writer.as_default():
                    self._val(val_dl,  batch_size, repeat, epoch)
        self.train_loss.reset_states()
        self.val_loss.reset_states()

    def _val_one_step(self, x, y):
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

    def _val(self, ds,  batch_size, repeat, epoch):
        pbar = tqdm(total=len(ds))
        pbar.set_description(f"Epoch: {epoch}. validation")
        for x, y in ds(batch_size, repeat):
            loss, _ = self._val_one_step(x, y)
            self.val_loss.update_state(loss)
            pbar.update(1)
            pbar.set_postfix(loss=float(self.val_loss.result()))
        pbar.close()

    def predict(self, x):
        logits = self.model(x, training=False)
        return logits

    def save_model(self, path, name):
        r"""Saves model in .h5 format

        Args:
            path (str)          : path of the folder where model is to be saved.
            name (str,optional) : name of the model.Default None. 
            """
        if name is None:
            name = ""
        name = time.localtime()+name

        self.model.save(path+name+".h5")
        print("Model saved at "+path + " as " + name+'.h5')

    def load_model(self, path):
        r"""Loads presaved model of .h5 format.

        Args:
            path (str) : path of the model to be loaded.

        """
        self.model = tf.keras.models.load_data(path)
        print("Model Loaded")
