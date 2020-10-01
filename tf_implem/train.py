import tensorflow as tf
import time
from tqdm import tqdm


class train():
    def __init__(self, model, criterion=None, weights=[1, 700], optimizer=None, checkpoint_path=None, max_to_keep=5):
        self.model = model
        if optimizer == None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.optimizer = optimizer
        self.steps = 0
        self.val_logits = []
        if criterion = None:
            self.criterion = tf.nn.softmax_cross_entropy_with_logits
        self.criterion = criterion
        self.weights = weights

        localtime = time.asctime()
        if checkpoint_path == None:
            checkpoint_path = "tf_ckpts/{}".format(localtime)
        self.checkpoint = tf.train.Checkpoint(
            model=self.model, optimizer=self.optimizer)

        self.manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=checkpoint_path,
            max_to_keep=5)

        self.train_summary_writer = tf.summary.create_file_writer(
            f"logs/{localtime}/train")
        self.val_summary_writer = tf.summary.create_file_writer(
            f"logs/{localtime}/val")
        self.train_loss = tf.keras.metrics.Mean(name="loss", dtype=tf.float32)
        self.val_loss = tf.keras.metrics.Mean(name="loss", dtype=tf.float32)

    def train_one_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.criterion(y, logits)
            loss, _ = tf.nn.weighted_moments(
                loss, (1, 2), np.sum(y*self.weights, axis=3))

        grads = tape.gradient(loss[0], self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))
        return loss[0]

    def _train_(self, ds, batch_size, epoch):
        pbar = tqdm(total=len(ds))
        pbar.set_description(f"Epoch: {epoch}. Traininig")

        for (x, y) in enumerate(ds()):
            loss = self.train_one_step(x, y)
            self.train_loss.update_state(loss)
            pbar.update(1)
            pbar.set_postfix(loss=float(self.train_loss.result()))
        pbar.close()

    def train(self, train_ds, val_ds=None, batch_size=4, epochs=10, save_checkpoints=True, checkpoint_freq=5, save_logits=False):
        self.checkpoint.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
        else:
            print("Initializing from scratch")

        for epoch in range(1, epochs + 1):

            with self.train_summary_writer.as_default():
                self._train_(train_ds, batch_size, epoch)
            if epoch % checkpoint_freq == 0:
                checkpoint_path = self.manager.save(self.optimizer.iterations)
                print("Model saved to {}".format(checkpoint_path))
            if val_ds is not None:
                with self.val_summary_writer.as_default():
                    self._val(val_ds, epoch, save_logits)
        self.train_loss.reset_states()
        self.val_loss.reset_states()

    def val_one_step(self, x, y):
        logits = self.model(x, training=False)
        loss = self.criterion(y, logits)
        loss, _ = tf.nn.weighted_moments(
            loss, (1, 2), np.sum(y*self.weights, axis=3))

        return loss[0], logits

    def _val(self, ds, epoch, save_logits=False):
        pbar = tqdm(total=len(ds))
        pbar.set_description(f"Epoch: {epoch}. validation")
        for (x, y) in enumerate(ds()):
            loss, logits = self.val_one_step(x, y)
            self.val_loss.update_state(loss)
            if save_logits:
                self.val_logits.append(logits)
            pbar.update(1)
            pbar.set_postfix(loss=float(self.val_loss.result()))
        pbar.close()

    def predict(self, x):
        logits = self.model(x, training=False)
        return logits
