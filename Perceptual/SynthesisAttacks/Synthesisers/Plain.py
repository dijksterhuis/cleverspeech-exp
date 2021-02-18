import tensorflow as tf

from experiments.Perceptual.SynthesisAttacks.Synthesisers.Base import Synth


class Plain(Synth):
    def __init__(self, batch):

        self.deltas = tf.Variable(
            tf.zeros([batch.size, batch.audios.max_length], dtype=tf.float32),
            trainable=True,
            validate_shape=True,
            dtype=tf.float32,
            name='qq_delta'
        )

        super().__init__()
        super().add_opt_vars(self.deltas)

    def synthesise(self):
        return self.deltas
