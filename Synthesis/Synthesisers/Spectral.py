import tensorflow as tf
import numpy as np

from Synthesis.Synthesisers.Base import Synth


class STFT(Synth):
    def __init__(self, batch, frame_step: int = 512, frame_length: int = 1024):

        # Tensorflow cannot optimise through the inverse stft if the windows are
        # not overlapping.
        assert frame_length > frame_step

        self.batch_size = batch_size = batch.size
        self.maxlen = maxlen = max(map(len, batch.audios.padded_audio))

        self.frame_length = int(frame_length)
        self.frame_step = int(frame_step)

        self.stft_deltas = None
        self.inverse_delta = None

        self.real_deltas = tf.Variable(
            np.zeros((batch_size, (maxlen // frame_step) - 1, frame_step), dtype=np.float32),
            trainable=True,
            validate_shape=True,
            name='qq_realdelta'
        )
        self.im_deltas = tf.Variable(
            np.zeros((batch_size, (maxlen // frame_step) - 1, frame_step), dtype=np.float32),
            trainable=True,
            validate_shape=True,
            name='qq_imdelta'
        )
        self.window_fn = tf.signal.inverse_stft_window_fn(self.frame_step)

        super().__init__()
        super().add_opt_vars(self.real_deltas, self.im_deltas)

    def synthesise(self):
        self.stft_deltas = tf.complex(
            self.real_deltas,
            self.im_deltas
        )

        self.inverse_delta = tf.signal.inverse_stft(
            self.stft_deltas,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            window_fn=self.window_fn
        )

        if (self.maxlen - self.inverse_delta.shape[1]) >= 0:

            pad_length = self.maxlen - self.inverse_delta.shape[1]

            padded = tf.concat(
                [self.inverse_delta,
                 np.zeros([self.batch_size, pad_length], dtype=np.float32)],
                axis=1
            )

        else:
            raise Exception(
                "Inversed STFT Delta is bigger than the Max audio length and I have no code to deal with this!"
            )

        return padded


class FramedDFT(Synth):
    def __init__(self, batch, frame_step, frame_length):

        # TODO: This will be used to perform spectral synthesis with
        #       non-overlapping frames
        super().__init__()
        raise NotImplementedError()
