import tensorflow as tf
import numpy as np

from experiments.Synthesis.Synthesisers.Base import Synth
from cleverspeech.utils.Utils import np_one


class Additive(Synth):
    @staticmethod
    def _exp_sigmoid(x, exponent=10.0, max_value=2.0, threshold=1e-7):
        x = tf.cast(x, dtype=tf.float32)
        return max_value * tf.nn.sigmoid(x) ** tf.math.log(exponent) + threshold

    @staticmethod
    def _upsample_amplitude(inputs, maxlen):

        """
        Taken directly from the magenta/DDSP library with minor modification for
        readability.

        :param inputs: the amplitude in [b, n_frames, n_osc]
        :param maxlen: max number of samples of the output audio (padded length)
        :return: interpolated amplitude values over time
        """

        n_frames = int(inputs.shape[1])
        n_intervals = (n_frames - 1)

        hop_size = maxlen // n_intervals
        win_len = 2 * hop_size
        window = tf.signal.hann_window(win_len)

        x = tf.transpose(inputs, perm=[0, 2, 1])
        x = x[:, :, :, tf.newaxis]
        window = window[tf.newaxis, tf.newaxis, tf.newaxis, :]
        x_windowed = (tf.cast(x, dtype=tf.float32) * window)
        x = tf.signal.overlap_and_add(x_windowed, hop_size)

        x = tf.transpose(x, perm=[0, 2, 1])
        x = x[:, hop_size:-hop_size, :]
        c = tf.concat(
            [x, tf.zeros([x.shape[0], maxlen - x.shape[1], x.shape[2]])],
            axis=1
        )
        return c

    @staticmethod
    def _harmonic_distribution(v, space=None, start=None, end=None, n_osc=None):
        """
        Create harmonically distributed tenors from fundamentals.
        Derived from the megenta/ddsp library.

        :param v: input vector of fundamental data
        :param n_osc: number of harmonics
        :param space: which space to use (linear, log, geometric etc.)
        :return: Harmonically distributed tensors
        """
        harmonic_dist = space(start, end, n_osc)
        harmonic_dist = harmonic_dist[tf.newaxis, tf.newaxis, :]
        return v * harmonic_dist

    @staticmethod
    def _resize(v, maxlen, method):
        """
        Re-sample freqs from n_frames to n_timesteps.
        Derived from the megenta/ddsp library.

        :param v: the values (freq or amp) to resize [bs, n_frames, n_osc]
        :param maxlen: the length of the resulting audio
        :param method: Bilinear vs. bicubic etc.
        :return: resampled tensors [batch, maxlen, n_osc]
        """
        # TODO -- tf.image.ResizeMethod.BICUBIC uses CPU rather than GPU
        tmp = v[:, :, tf.newaxis, :]
        resampled = tf.image.resize(tmp, [maxlen, v.shape[-1]], method=method)
        return resampled[:, :, 0, :]

    @staticmethod
    def _gen_mask(actlen, maxlen):
        """
        Generate the delta masks to zero out parts of the vector that correspond
        to padded sections of input audio

        :param actlen: actual length of the original example in samples
        :param maxlen: maximum length in the batch of original examples
        :return: masked delta
        """
        for l in actlen:
            m = list()
            for i in range(maxlen):
                if i < l:
                    m.append(1)
                else:
                    m.append(0)
            yield m

    @staticmethod
    def _gen_f0_hz(batch_size, n_frames, starting_f0_hz):
        """
        Initialise dependent frequency variables.

        :param batch_size: size of batch data.
        :param n_frames: number of samples (this is maxlen)
        :param starting_f0_hz: initial fundamental frequency.
        :return: vector of the fundamental frequency for each example in a batch
                -- shape [b, n_frames].
        """
        for _ in range(batch_size):
            b = list()
            for _ in range(n_frames):
                b.append([starting_f0_hz])
            yield b

    @staticmethod
    def _gen_hz(batch_size, n_frames, n_osc, starting_f0_hz=220):
        """
        Initialise independent frequency variables.

        :param batch_size: size of batch data.
        :param n_frames: number of samples (this is maxlen).
        :param n_osc: number of totoal oscillators.
        :param starting_f0_hz: initial fundamental frequency.
        :return: vector of the fundamental frequency for each example in a batch
                -- shape [b, n_frames, n_osc].
        """
        for _ in range(batch_size):
            a = list()
            rand = np.random.uniform(0.1, 1.0, n_osc)
            for _ in range(n_frames):
                # start somewhat harmonically, otherwise the frequencies values
                # stick around the same position
                halfway_harmonic = np.linspace(float(1), float(n_osc), n_osc)
                flat = np.linspace(float(1), float(1), n_osc)
                # then uniformly randomise it by scaling the frequencies up or
                # down by factor of 2

                random_harmonic = starting_f0_hz * halfway_harmonic * rand
                a.append(starting_f0_hz * flat * rand)
            yield a

    def __init__(
            self,
            freq_deltas,
            phase_deltas,
            amplitude_deltas,
            n_osc=None,
            waveform=tf.sin,
            sample_rate=None,
            maxlen=None,
            actlen=None,
            normalise=True,
            alpha=1000,
    ):

        self.n_osc = n_osc
        self.sample_rate = sample_rate
        self.maxlen = maxlen
        self.actlen = actlen
        self.waveform = waveform
        self.normalise = normalise

        self.freq_deltas = freq_deltas
        self.phase_deltas = phase_deltas
        self.amplitude_deltas = amplitude_deltas
        self.alpha = alpha
        self.deltas = None

        super().__init__()
        super().add_opt_vars(self.freq_deltas, self.amplitude_deltas, )  # self.phase_deltas)

    def synthesise(self, freqs=None, amps=None, phases=None):
        """
        Do Additive Synthesis.

        Derived from the megenta/ddsp library.

        :param freqs: Frequencies in Hz to synthesise -- [b, n_frames, n_osc]
        :param amps: Amplitudes for each frequency -- [b, n_frames, n_osc]
        :return: Synthesised audio vectors (type float32) -- [b, n_frames]
        """

        # == zero signal above Nyquist frequency
        # amps = tf.where(
        #     tf.greater_equal(freqs, float(self.sample_rate / 2)),
        #     tf.zeros_like(amps),
        #     amps
        # )
        #
        # # == null signal when a frequency is less than 0 Hz
        # amps = tf.where(
        #     tf.less_equal(freqs, float(0)),
        #     tf.zeros_like(amps),
        #     amps
        # )

        # add a new empty time dimension so phases can add to freqs
        #phases = phases[:, tf.newaxis, :]

        # == limit between -1 <= x <= +1, but use tanh so we never stray into an
        # clipped values and stay there.
        #phases = tf.tanh(phases)
        #
        # # == limit phases within -1 <= phi <= 1
        # phases = tf.where(
        #     tf.less_equal(phases, float(-1)),
        #     tf.zeros_like(phases),
        #     phases
        # )
        # phases = tf.where(
        #     tf.greater_equal(phases, float(1)),
        #     tf.zeros_like(phases),
        #     phases
        # )

        # TODO: Convert amplitude to Decibel scale -- 20.log10(A) ?
        # https://github.com/tensorflow/tensorflow/issues/1666
        # log_amps = tf.log(amps)/ tf.log(tf.constant(10, dtype=amps.dtype))
        # amps = 20.0 * log_amps

        # == interpolation between frames

        # Interpolate from [b, n_frames, n_osc] -> [b, n_samples, n_osc]
        # f0_t = [b, n_frames], Hz
        # p_t = [b, n_frames, n_osc], Hz
        # a_t = [b, n_frames, n_osc], ?

        # TODO: BICUBIC GPU Implementation for frequency interpolation.
        # TODO: CHeck McAuley et al. H+N interpolation of phase.

        method = tf.image.ResizeMethod.BILINEAR
        f_t_hz = self._resize(freqs, self.maxlen, method)
        a_t = self._upsample_amplitude(amps, self.maxlen)

        # == convert fundamental frequency from Hz to rad/s
        f_t = f_t_hz * float((np.pi * 2))
        f_t = f_t / float(self.sample_rate)

        # == Additive Synthesis!
        w_t = tf.cumsum(f_t, axis=1) # + phases, axis=1)
        y_t = a_t * self.waveform(w_t)

        # == Sum and normalise if required
        if self.normalise:
            self.deltas = tf.reduce_sum(y_t, -1) / tf.constant(float(self.n_osc))
        else:
            self.deltas = tf.reduce_sum(y_t, -1)
        return self.deltas


class FullyHarmonic(Additive):
    def __init__(
            self,
            batch,
            n_osc=32,
            frame_length=128,
            frame_step=128,
            sample_rate=16000,
            normalise=True,
            initial_hz=440,
            initial_amp=0,
    ):
        """
        Attack Graph for Harmonic Additive Synthesis.

        :param n_osc: number of oscillators in the oscillator bank
        :param frame_step: the window step size for initial harmonic content
        :param frame_length: the window length for initial harmonic content
        :param sample_rate: sample rate of the original audio
        :param batch: batch of examples from the BatchFactory
        :param normalise: whether to normalise the oscillator bank or not.
        :param initial_hz: initial frequency for fundamental oscillator
        :param initial_amp: initial amplitude for *all* oscillators
        :param alpha: scaling factor for amplitude values
        """

        batch_size = batch.size
        maxlen = batch.audios.max_length
        actual_lengths = batch.audios.actual_lengths

        self.maxlen = maxlen
        self.n_osc = n_osc
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.sample_rate = sample_rate
        self.normalise = normalise

        self.n_frames = n_frames = ((maxlen - frame_length) // frame_step) + 1

        # batch_size * time * 1
        f0 = np.array(
            [f for f in self._gen_f0_hz(batch_size, n_frames, initial_hz)],
            dtype=np.float32
        )
        # batch_size * time * 1
        a0 = np.array(
            [f for f in self._gen_f0_hz(batch_size, n_frames, initial_amp)],
            dtype=np.float32
        )
        freq_deltas = tf.Variable(
            f0,
            trainable=True,
            validate_shape=True,
            dtype=tf.float32,
            name='qq_freq'
        )

        phase_deltas = tf.Variable(
            tf.zeros((batch_size, n_osc)),
            trainable=True,
            validate_shape=True,
            dtype=tf.float32,
            name='qq_phase'
        )

        amplitude_deltas = tf.Variable(
            a0,
            trainable=True,
            validate_shape=True,
            dtype=tf.float32,
            name='qq_amp'
        )

        super().__init__(
            freq_deltas,
            phase_deltas,
            amplitude_deltas,
            n_osc=self.n_osc,
            sample_rate=sample_rate,
            maxlen=maxlen,
            actlen=actual_lengths,
            normalise=normalise,
        )

    def synthesise(self, freqs=None, amps=None, phases=None):

        freqs = self.freq_deltas if freqs is None else freqs
        phases = self.phase_deltas if phases is None else phases
        amps = (self.amplitude_deltas if amps is None else amps)

        # == harmonic distribution of frequencies and amplitudes

        freq_frames_hz = self._harmonic_distribution(
            freqs,
            space=tf.linspace,
            start=float(1),
            end=float(self.n_osc),
            n_osc=self.n_osc
        )

        amp_frames = self._harmonic_distribution(
            amps,
            space=tf.linspace,  # np.logspace,
            start=float(self.n_osc),  # float(1),
            end=float(1),  # float(0.1),
            n_osc=self.n_osc
        )

        return super().synthesise(freqs=freq_frames_hz, amps=amp_frames, phases=phases)


class FreqHarmonic(Additive):
    def __init__(
            self,
            batch,
            n_osc=32,
            frame_length=128,
            frame_step=128,
            sample_rate=16000,
            normalise=True,
            waveform=tf.sin,
            initial_hz=440,
            initial_amp=0,
    ):
        """
        Attack Graph for Harmonic Additive Synthesis.

        :param n_osc: number of oscillators in the oscillator bank.
        :param frame_step: the window step size for initial harmonic content.
        :param frame_length: the window length for initial harmonic content.
        :param sample_rate: sample rate of the original audio.
        :param batch: batch data from the BatchFactory.
        :param normalise: whether to normalise the oscillator bank or not.
        :param initial_hz: initial frequency for fundamental oscillator.
        :param initial_amp: initial amplitude for *all* oscillators.
        :param alpha: scaling factor for amplitude values
        """

        batch_size = batch.size
        maxlen = batch.audios.max_length
        actual_lengths = batch.audios.actual_lengths

        self.maxlen = maxlen
        self.n_osc = n_osc
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.sample_rate = sample_rate
        self.normalise = normalise

        self.n_frames = n_frames = ((maxlen - frame_length) // frame_step) + 1

        # batch_size * time * 1
        f0 = np.array(
            [f for f in self._gen_f0_hz(batch_size, n_frames, initial_hz)],
            dtype=np.float32
        )
        freq_deltas = tf.Variable(
            f0,
            trainable=True,
            validate_shape=True,
            dtype=tf.float32,
            name='qq_freq'
        )

        phase_deltas = tf.Variable(
            tf.zeros((batch_size, n_osc)),
            trainable=True,
            validate_shape=True,
            dtype=tf.float32,
            name='qq_phase'
        )

        amplitude_deltas = tf.Variable(
            initial_amp * np_one((batch_size, n_frames, n_osc), np.float32),
            trainable=True,
            validate_shape=True,
            dtype=tf.float32,
            name='qq_amp'
        )

        super().__init__(
            freq_deltas,
            phase_deltas,
            amplitude_deltas,
            n_osc=n_osc,
            sample_rate=sample_rate,
            waveform=waveform,
            maxlen=maxlen,
            actlen=actual_lengths,
            normalise=normalise
        )

    def synthesise(self, freqs=None, amps=None, phases=None):

        freqs = self.freq_deltas if freqs is None else freqs
        phases = self.phase_deltas if phases is None else phases
        amps = (self.amplitude_deltas if amps is None else amps)

        # == harmonic distribution of frequencies only

        freq_frames_hz = self._harmonic_distribution(
            freqs,
            space=np.linspace,  # np.logspace,
            start=float(1),  # float(1),
            end=self.n_osc,  # float(0.1),
            n_osc=self.n_osc
        )

        return super().synthesise(freqs=freq_frames_hz, amps=amps, phases=phases)


class InHarmonic(Additive):
    def __init__(
            self,
            batch,
            n_osc=32,
            frame_length=128,
            frame_step=128,
            sample_rate=16000,
            normalise=True,
            initial_hz=440,
            initial_amp=0,
    ):
        """
        Attack Graph for Inharmonic Additive Synthesis.

        :param n_osc: number of oscillators in the oscillator bank.
        :param frame_step: the window step size for initial harmonic content.
        :param frame_length: the window length for initial harmonic content.
        :param sample_rate: sample rate of the original audio.
        :param batch: batch data from the BatchFactory.
        :param normalise: whether to normalise the oscillator bank or not.
        :param initial_hz: initial frequency for *all* oscillators.
        :param initial_amp: initial amplitude for *all* oscillators.
        :param alpha: scaling factor for amplitude values
        """
        # == Sanity check inputs

        batch_size = batch.size
        maxlen = batch.audios.max_length
        actual_lengths = batch.audios.actual_lengths

        self.frame_length = frame_length
        self.frame_step = frame_step

        self.n_frames = n_frames = ((maxlen - frame_length) // frame_step) + 1

        # batch_size * time * n_osc
        f = np.array(
            [f for f in
             self._gen_hz(batch_size, n_frames, n_osc, initial_hz)],
            dtype=np.float32
        )

        freq_deltas = tf.Variable(
            f,
            trainable=True,
            validate_shape=True,
            dtype=tf.float32,
            name='qq_freq'
        )

        phase_deltas = tf.Variable(
            tf.zeros((batch_size, n_osc)),
            trainable=True,
            validate_shape=True,
            dtype=tf.float32,
            name='qq_phase'
        )

        amplitude_deltas = tf.Variable(
            initial_amp * np_one((batch_size, n_frames, n_osc), np.float32),
            trainable=True,
            validate_shape=True,
            dtype=tf.float32,
            name='qq_amp'
        )

        super().__init__(
            freq_deltas,
            phase_deltas,
            amplitude_deltas,
            n_osc=n_osc,
            sample_rate=sample_rate,
            maxlen=maxlen,
            actlen=actual_lengths,
            normalise=normalise,
            waveform=tf.cos
        )

    def synthesise(self, freqs=None, amps=None, phases=None):

        # == no harmonic distribution

        freqs = self.freq_deltas if freqs is None else freqs
        phases = self.phase_deltas if phases is None else phases
        amps = (self.amplitude_deltas if amps is None else amps)

        return super().synthesise(freqs=freqs, amps=amps, phases=phases)

