from experiments.Perceptual.SynthesisAttacks.Synthesisers.Base import Synth
from experiments.Perceptual.SynthesisAttacks.Synthesisers import Plain
from experiments.Perceptual.SynthesisAttacks.Synthesisers import Additive


class FreqHarmonicPlusPlain(Synth):
    def __init__(self, batch, frame_length=512, frame_step=512, n_osc=16, initial_hz=440, noise_weight=0.5, normalise=False):

        assert type(noise_weight) is float
        assert 0.0 <= noise_weight <= 1.0

        self.det = Additive.FreqHarmonic(
            batch,
            frame_length=frame_length,
            frame_step=frame_step,
            n_osc=n_osc,
            initial_hz=initial_hz,
            normalise=normalise,
        )
        self.noise = Plain.Plain(batch)
        self.noise_weight = float(noise_weight)

        super().__init__()
        super().add_opt_vars(*self.det.opt_vars, *self.noise.opt_vars)

    def synthesise(self):

        weighted_det = (1.0 - self.noise_weight) * self.det.synthesise()
        weighted_noise = self.noise_weight * self.noise.synthesise()

        return weighted_det + weighted_noise


class FullyHarmonicPlusPlain(Synth):
    def __init__(self, batch, frame_length=512, frame_step=512, n_osc=16, initial_hz=440, noise_weight=0.5, normalise=False):

        assert type(noise_weight) is float
        assert 0.0 <= noise_weight <= 1.0

        self.det = Additive.FullyHarmonic(
            batch,
            frame_length=frame_length,
            frame_step=frame_step,
            n_osc=n_osc,
            initial_hz=initial_hz,
            normalise=normalise,
        )
        self.noise = Plain.Plain(batch)
        self.noise_weight = float(noise_weight)

        super().__init__()
        super().add_opt_vars(*self.det.opt_vars, *self.noise.opt_vars)

    def synthesise(self):
        weighted_det = (1.0 - self.noise_weight) * self.det.synthesise()
        weighted_noise = self.noise_weight * self.noise.synthesise()

        return weighted_det + weighted_noise


class InharmonicPlusPlain(Synth):
    def __init__(self, batch, frame_length=512, frame_step=512, n_osc=16, initial_hz=440, noise_weight=0.5, normalise=False):

        assert type(noise_weight) is float
        assert 0.0 <= noise_weight <= 1.0

        self.det = Additive.InHarmonic(
            batch,
            frame_length=frame_length,
            frame_step=frame_step,
            n_osc=n_osc,
            initial_hz=initial_hz,
            normalise=normalise,
        )
        self.noise = Plain.Plain(batch)
        self.noise_weight = float(noise_weight)

        super().__init__()
        super().add_opt_vars(*self.det.opt_vars, *self.noise.opt_vars)

    def synthesise(self):
        weighted_det = (1.0 - self.noise_weight) * self.det.synthesise()
        weighted_noise = self.noise_weight * self.noise.synthesise()

        return weighted_det + weighted_noise

