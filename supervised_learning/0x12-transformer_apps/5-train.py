#!/usr/bin/env python3
""""""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """"""
    def __init__(self, dm, warmup_steps=4000):
        """"""
        super(CustomSchedule, self).__init__()

        self.dm = dm
        self.dm = tf.cast(self.dm, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """"""
    learning_rate = CustomSchedule(dm)

    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)

    temp_learning_rate_schedule = CustomSchedule(dm)

    loss_object = (tf.keras.losses
                   .SparseCategoricalCrossentropy(from_logits=True,
                                                  reduction='none'))

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

    def accuracy_function(real, pred):
        accuracies = tf.equal(real, tf.argmax(pred, axis=2))

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    data = Dataset(batch_size, max_len)
    encoder = data.tokenizer_pt
    decoder = data.tokenizer_en

    transformer = Transformer(N,
                              dm,
                              h,
                              hidden,
                              max_len,
                              max_len,
                              max_len,
                              max_len,
                              encoder,
                              decoder
                              )
    checkpoint_path = "./checkpoints/train"

    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt,
                                              checkpoint_path,
                                              max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    train_step_signature = [
                            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                            tf.TensorSpec(shape=(None, None), dtype=tf.int64),
                            ]

    # @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        (enc_padding_mask,
         combined_mask,
         dec_padding_mask) = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = transformer(inp,
                                         tar_inp,
                                         True,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)
            loss = loss_function(tar_real, predictions)

            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients,
                                          transformer.trainable_variables))

            train_loss(loss)
            train_accuracy(accuracy_function(tar_real, predictions))

    train_dataset = data.data_train  # .batch(batch_size).repeat()

    for epoch in range(epochs):

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(train_dataset):
            train_step(inp, tar)

            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'
                      .format(epoch + 1, batch, train_loss.result(),
                              train_accuracy.result()))

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                            ckpt_save_path))

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                        train_loss.result(),
                                                        train_accuracy.result()
                                                        ))
