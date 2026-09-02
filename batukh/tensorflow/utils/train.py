"""Custom TensorFlow training loop shared by segmentation and OCR models."""

import math
import os
from datetime import datetime, timezone
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tqdm.auto import tqdm


class Train:
    """Train, validate, checkpoint, and run a custom Keras model."""

    def __init__(self, model, is_ocr=False, blank_index=None, log_dir="logs"):
        self.is_ocr = bool(is_ocr)
        self.model = model
        self.blank_index = (
            model.n_classes - 1 if blank_index is None else int(blank_index)
        )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        run_dir = Path(log_dir) / self.__class__.__name__ / timestamp
        self.train_summary_writer = tf.summary.create_file_writer(
            str(run_dir / "train")
        )
        self.val_summary_writer = tf.summary.create_file_writer(str(run_dir / "val"))
        self.train_loss = keras.metrics.Mean(name="train_loss", dtype=tf.float32)
        self.val_loss = keras.metrics.Mean(name="val_loss", dtype=tf.float32)
        self.history = {"loss": [], "val_loss": []}

        self.optimizer = None
        self.criterion = None
        self.class_weights = None
        self.checkpoint = None
        self.manager = None

    def _compute_loss(self, labels, logits):
        if self.is_ocr:
            logit_length = None
            if isinstance(labels, (tuple, list)):
                labels, logit_length = labels
            if logit_length is None:
                logit_length = tf.fill(
                    (tf.shape(logits)[0],),
                    tf.shape(logits)[1],
                )
            else:
                logit_length = tf.minimum(
                    tf.cast(logit_length, tf.int32),
                    tf.shape(logits)[1],
                )
            if self.criterion is None:
                loss = tf.nn.ctc_loss(
                    labels=labels,
                    logits=logits,
                    label_length=None,
                    logit_length=logit_length,
                    logits_time_major=False,
                    blank_index=self.blank_index,
                )
            else:
                loss = self.criterion(labels, logits, logit_length)
            loss = tf.reduce_mean(loss)
        else:
            per_pixel_loss = self.criterion(labels, logits)
            pixel_weights = tf.reduce_sum(
                labels * self.class_weights,
                axis=-1,
            )
            loss = tf.math.divide_no_nan(
                tf.reduce_sum(per_pixel_loss * pixel_weights),
                tf.reduce_sum(pixel_weights),
            )

        if self.model.losses:
            loss += tf.add_n(self.model.losses)
        return loss

    def _train_one_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            logits = self.model(inputs, training=True)
            loss = self._compute_loss(labels, logits)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients_and_variables = [
            (gradient, variable)
            for gradient, variable in zip(
                gradients,
                self.model.trainable_variables,
            )
            if gradient is not None
        ]
        if not gradients_and_variables:
            raise RuntimeError("No gradients were produced for the model.")
        self.optimizer.apply_gradients(gradients_and_variables)
        return loss

    def _val_one_step(self, inputs, labels):
        logits = self.model(inputs, training=False)
        return self._compute_loss(labels, logits), logits

    @staticmethod
    def _make_dataset(loader, batch_size, repeat, shuffle):
        if isinstance(loader, tf.data.Dataset):
            return loader
        if not callable(loader):
            raise TypeError("A loader must be callable or a tf.data.Dataset.")
        return loader(
            batch_size=batch_size,
            repeat=repeat,
            shuffle=shuffle,
        )

    @staticmethod
    def _step_count(loader, batch_size, repeat):
        if hasattr(loader, "__len__"):
            return math.ceil(len(loader) / batch_size) * repeat
        return None

    @staticmethod
    def _should_log(step, log_freq):
        return log_freq is not None and log_freq > 0 and step % log_freq == 0

    def _write_summary(self, writer, split, inputs, labels, loss):
        step = int(self.optimizer.iterations.numpy())
        with writer.as_default():
            tf.summary.scalar("loss", loss, step=step)
            if not self.is_ocr:
                logits = self.model(inputs[:1], training=False)
                prediction = tf.cast(
                    tf.argmax(logits, axis=-1)[..., tf.newaxis],
                    tf.float32,
                )
                target = tf.cast(
                    tf.argmax(labels[:1], axis=-1)[..., tf.newaxis],
                    tf.float32,
                )
                scale = tf.cast(tf.maximum(self.model.n_classes - 1, 1), tf.float32)
                comparison = tf.concat((prediction / scale, target / scale), axis=2)
                tf.summary.image(f"predictions/{split}", comparison, step=step)

    def _train(self, loader, epoch, batch_size, repeat, log_freq):
        dataset = self._make_dataset(loader, batch_size, repeat, shuffle=True)
        total = self._step_count(loader, batch_size, repeat)
        self.train_loss.reset_state()

        progress = tqdm(total=total, desc=f"Epoch {epoch}: training")
        for inputs, labels in dataset:
            loss = self._train_one_step(inputs, labels)
            self.train_loss.update_state(loss)
            step = int(self.optimizer.iterations.numpy())
            if self._should_log(step, log_freq):
                self._write_summary(
                    self.train_summary_writer,
                    "train",
                    inputs,
                    labels,
                    self.train_loss.result(),
                )
            progress.update(1)
            progress.set_postfix(loss=float(self.train_loss.result().numpy()))
        progress.close()
        return float(self.train_loss.result().numpy())

    def _val(self, loader, epoch, batch_size, repeat, log_freq):
        dataset = self._make_dataset(loader, batch_size, repeat, shuffle=False)
        total = self._step_count(loader, batch_size, repeat)
        self.val_loss.reset_state()

        progress = tqdm(total=total, desc=f"Epoch {epoch}: validation")
        for inputs, labels in dataset:
            loss, _ = self._val_one_step(inputs, labels)
            self.val_loss.update_state(loss)
            step = int(self.optimizer.iterations.numpy())
            if self._should_log(step, log_freq):
                self._write_summary(
                    self.val_summary_writer,
                    "validation",
                    inputs,
                    labels,
                    self.val_loss.result(),
                )
            progress.update(1)
            progress.set_postfix(loss=float(self.val_loss.result().numpy()))
        progress.close()
        return float(self.val_loss.result().numpy())

    def train(
        self,
        n_epochs,
        train_dl=None,
        val_dl=None,
        batch_size=1,
        repeat=1,
        criterion=None,
        class_weights=None,
        optimizer=None,
        weight_decay=None,
        learning_rate=0.0001,
        lr_decay=None,
        save_checkpoints=True,
        checkpoint_freq=None,
        checkpoint_path=None,
        max_to_keep=5,
        log_freq=100,
    ):
        if train_dl is None:
            raise ValueError("train_dl is required; call load_data() or pass a loader.")
        if n_epochs < 1:
            raise ValueError("n_epochs must be at least 1.")
        if batch_size < 1 or repeat < 1:
            raise ValueError("batch_size and repeat must be at least 1.")

        if lr_decay is not None:
            learning_rate = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=100_000,
                decay_rate=lr_decay,
                staircase=True,
            )

        if optimizer is None:
            if weight_decay is not None and weight_decay > 0:
                optimizer = keras.optimizers.AdamW(
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                )
            else:
                optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.optimizer = optimizer

        if self.is_ocr:
            self.criterion = criterion
        else:
            self.criterion = criterion or keras.losses.CategoricalCrossentropy(
                from_logits=True,
                reduction="none",
            )

        if class_weights is None:
            class_weights = [1.0] * self.model.n_classes
        if len(class_weights) != self.model.n_classes:
            raise ValueError("class_weights must have one value per model class.")
        self.class_weights = tf.reshape(
            tf.convert_to_tensor(class_weights, dtype=tf.float32),
            (1, 1, 1, self.model.n_classes),
        )

        if checkpoint_freq is None:
            checkpoint_freq = n_epochs // 10 + 1
        if checkpoint_freq < 1:
            raise ValueError("checkpoint_freq must be at least 1.")

        self.manager = None
        if save_checkpoints:
            if checkpoint_path is None:
                checkpoint_path = Path("tf_ckpts") / self.__class__.__name__
            self.checkpoint = tf.train.Checkpoint(
                model=self.model,
                optimizer=self.optimizer,
            )
            self.manager = tf.train.CheckpointManager(
                self.checkpoint,
                directory=str(checkpoint_path),
                max_to_keep=max_to_keep,
            )
            if self.manager.latest_checkpoint:
                self.checkpoint.restore(self.manager.latest_checkpoint)
                print(f"Restored from {self.manager.latest_checkpoint}")
            else:
                print("Initializing from scratch")

        self.history = {"loss": [], "val_loss": []}
        for epoch in range(1, n_epochs + 1):
            train_result = self._train(
                train_dl,
                epoch,
                batch_size,
                repeat,
                log_freq,
            )
            self.history["loss"].append(train_result)

            if val_dl is not None:
                val_result = self._val(
                    val_dl,
                    epoch,
                    batch_size,
                    repeat,
                    log_freq,
                )
                self.history["val_loss"].append(val_result)

            if self.manager is not None and epoch % checkpoint_freq == 0:
                saved_path = self.manager.save(
                    checkpoint_number=int(self.optimizer.iterations.numpy())
                )
                print(f"Model saved to {saved_path}")

        self.train_summary_writer.flush()
        self.val_summary_writer.flush()
        return self.history

    def predict(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        if inputs.shape.rank == 3:
            inputs = inputs[tf.newaxis, ...]
        if inputs.shape.rank != 4:
            raise ValueError("predict expects an HWC image or a batch of HWC images.")
        return self.model(inputs, training=False)

    def save_model(self, path, name=None):
        raw_path = os.fspath(path)
        destination = Path(path)
        if name is not None:
            destination = destination / name
        elif destination.is_dir() or raw_path.endswith((os.sep, "/")):
            destination = destination / "model"
        if not str(destination).endswith(".weights.h5"):
            destination = Path(f"{destination}.weights.h5")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(destination)
        print(f"Model saved at {destination}")
        return destination

    def load_model(self, path):
        source = Path(path)
        legacy_index = Path(f"{source}.index")
        if legacy_index.exists():
            checkpoint = tf.train.Checkpoint(model=self.model)
            checkpoint.restore(str(source)).expect_partial()
            print(f"Legacy TensorFlow checkpoint loaded from {source}")
            return source
        if not source.exists() and not str(source).endswith(".weights.h5"):
            candidate = Path(f"{source}.weights.h5")
            if candidate.exists():
                source = candidate
        self.model.load_weights(source)
        print(f"Model loaded from {source}")
        return source
