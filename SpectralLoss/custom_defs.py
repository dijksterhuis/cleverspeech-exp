import tensorflow as tf
import numpy as np


class SpectralLoss(object):
    def __init__(self, attack_graph, frame_size: int = 512, norm: int = 2, loss_weight: float = 10.0e-7):

        self.spectrogram_diff = tf.signal.stft(
            signals=attack_graph.placeholders.audios - attack_graph.bounded_deltas,
            frame_length=int(frame_size),
            frame_step=int(frame_size * 2),
            fft_length=int(frame_size),
            pad_end=True
        )

        self.magnitude_diff = tf.cast(tf.abs(self.spectrogram_diff), tf.float32)

        self.mag_loss_fn = tf.reduce_mean(self.magnitude_diff ** norm)

        self.loss_fn = loss_weight * self.mag_loss_fn


class NormalisedSpectralLoss(object):
    def __init__(self, attack_graph, frame_size: int = 128, overlap: float = 0.75, norm: int = 2, loss_weight: float = 100.0):

        self.spectrogram_orig = tf.signal.stft(
            signals=attack_graph.placeholders.audios,
            frame_length=int(frame_size),
            frame_step=int(frame_size * 2),
            fft_length=int(frame_size),
            pad_end=False
        )

        self.spectrogram_delta = tf.signal.stft(
            signals=attack_graph.bounded_deltas,
            frame_length=int(frame_size),
            frame_step=int(frame_size * 2 * overlap),
            fft_length=int(frame_size),
            pad_end=False
        )

        self.magnitude_orig = tf.cast(tf.abs(self.spectrogram_orig), tf.float32)
        self.magnitude_delta = tf.cast(tf.abs(self.spectrogram_delta), tf.float32)

        self.mag_norm_delta = tf.reduce_mean(self.magnitude_delta ** norm, axis=1)
        self.mag_norm_orig = tf.reduce_mean(self.magnitude_orig ** norm, axis=1)
        # dividing at the reduced mean stage avoids us accidentally dividing by
        # zero if we were to have a zero valued original frame.
        self.mag_loss_fn = self.mag_norm_delta / self.mag_norm_orig

        self.loss_fn = loss_weight * self.mag_loss_fn


class MultiScaleSpectralLoss(object):
    def __init__(self, attack_graph, frame_size=512, norm=1, loss_weight=1.0):

        self.spectrogram_diff = tf.signal.stft(
            signals=attack_graph.placeholders.audios - attack_graph.bounded_deltas,
            frame_length=int(frame_size),
            frame_step=int(frame_size * 2),
            fft_length=int(frame_size),
            pad_end=True
        )

        self.magnitude_diff = tf.cast(tf.abs(self.spectrogram_diff), tf.float32)
        self.log_magnitude_diff = tf.log(self.magnitude_diff + 10.0e-10)

        self.mag_loss_fn = tf.reduce_mean(self.magnitude_diff ** norm)
        self.log_mag_loss_fn = tf.reduce_mean(self.log_magnitude_diff ** norm)

        self.loss_fn = loss_weight * self.mag_loss_fn


