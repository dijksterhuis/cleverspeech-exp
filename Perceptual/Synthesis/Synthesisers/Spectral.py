import tensorflow as tf
import numpy as np

from experiments.Perceptual.Synthesis.Synthesisers.Base import Synth


class STFT(Synth):
    def __init__(self, batch, frame_step: int = 512, frame_length: int = 1024, fft_length: int = 1024):

        self.batch_size = batch_size = batch.size
        self.maxlen = maxlen = max(map(len, batch.audios["padded_audio"]))

        self.frame_length = int(frame_length)
        self.frame_step = int(frame_step)
        self.fft_length = int(fft_length)

        if self.frame_length == self.frame_step:
            assert self.fft_length > self.frame_step
            self.window_fn = None

        elif self.fft_length == self.frame_step:
            assert self.frame_length > self.frame_step
            self.window_fn = tf.signal.inverse_stft_window_fn(self.frame_step)

        else:
            raise ValueError(
                "Mismatched combination of frame length/frame step/fft length."
            )

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
        self.stft_deltas = tf.complex(
            self.real_deltas,
            self.im_deltas
        )

        super().__init__()
        super().add_opt_vars(self.real_deltas, self.im_deltas)

    def synthesise(self):

        self.inverse_delta = tf.signal.inverse_stft(
            self.stft_deltas,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            #window_fn=self.window_fn,
            fft_length=self.fft_length,

        )
        pad_length = self.maxlen - self.inverse_delta.shape[1]

        if pad_length > 0:
            padded = tf.concat(
                [self.inverse_delta,
                 np.zeros([self.batch_size, pad_length], dtype=np.float32)],
                axis=1
            )

        elif pad_length == 0:
            padded = self.inverse_delta

        else:
            raise NotImplementedError(
                "Inversed STFT Delta is bigger than the Max audio length and I have no code to deal with this!"
            )

        return padded
