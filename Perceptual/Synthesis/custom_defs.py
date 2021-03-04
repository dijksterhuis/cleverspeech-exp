import tensorflow as tf
import numpy as np

from cleverspeech.graph.Placeholders import Placeholders
from cleverspeech.graph.Procedures import UpdateOnDecoding, UpdateOnLoss
from cleverspeech.utils.Utils import np_arr, np_zero, np_one, lcomp


class SynthesisAttack:
    def __init__(self, sess, batch, hard_constraint, synthesiser):

        """
        Attack graph from 2018 Carlini & Wager Targeted Audio Attack.
        # TODO: norm should be a class defined in Audio.Distance
        # TODO: Distance classes should have `bound()` and `analyse()` methods

        :param sess: tf.Session which will be used in the attack.
        :param tau: the upper bound for the size of the perturbation
        :param batch: The batch of fata
        :param synthesis: the `Audio.Synthesis` class which generates the perturbation
        :param constraint: the `Audio.Distance` class being used for the attack

        """
        batch_size = batch.size
        max_len = batch.audios["max_samples"]
        act_lengths = batch.audios["n_samples"]

        self.placeholders = Placeholders(batch_size, max_len)

        self.masks = tf.Variable(
            tf.zeros([batch_size, max_len]),
            trainable=False,
            validate_shape=True,
            dtype=tf.float32,
            name='qq_masks'
        )

        self.synthesiser = synthesiser

        # Generate the delta synth parameter objects which we will optimise
        deltas = synthesiser.synthesise()

        # Mask deltas first so we zero value *any part of the signal* that is
        # zero value padded in the original audio
        deltas = deltas * self.masks

        # Restrict delta to valid space before applying constraints

        lower = -2.0 ** 15
        upper = 2.0 ** 15 - 1

        valid_deltas = tf.clip_by_value(
            deltas,
            clip_value_min=lower,
            clip_value_max=upper
        )

        self.final_deltas = hard_constraint.clip(valid_deltas)

        # clip example to valid range
        self.adversarial_examples = tf.clip_by_value(
            self.final_deltas + self.placeholders.audios,
            clip_value_min=lower,
            clip_value_max=upper
        )

        # initialise static variables
        initial_masks = np_arr(
            lcomp(self._gen_mask(act_lengths, max_len)),
            np.float32
        )

        sess.run(self.masks.assign(initial_masks))

        self.opt_vars = synthesiser.opt_vars


    @staticmethod
    def _gen_mask(lengths, max_len):
        """
        Generate the zero / one valued mask for the perturbations in a batch
        Mask value == 1 when actual < max, == 0 when actual > max

        :param lengths: number of samples per audio example in a batch
        :param max_len: maximum numb. of samples of audio examples in a batch
        :yield: 0/1 valued mask vector of length max_len
        """
        for l in lengths:
            m = list()
            for i in range(max_len):
                if i < l:
                    m.append(1)
                else:
                    m.append(0)
            yield m


class AdditiveAmplitudeLoss(object):
    def __init__(self, attack_graph, loss_weight=1.0):

        g, b = attack_graph, attack_graph.batch

        assert b.audios["ds_feats"].all() > 0

        self.loss_fn = tf.reduce_sum(
            tf.abs(g.synthesis.amplitude_deltas),
            axis=[1, 2]
        )
        self.loss_fn = self.loss_fn * loss_weight / b.audios["ds_feats"]


class AdditiveEnergyLoss(object):
    def __init__(self, attack_graph, loss_weight=1.0):

        g, b = attack_graph, attack_graph.batch

        assert b.audios["ds_feats"].all() > 0

        self.loss_fn = tf.reduce_sum(
            tf.square(g.synthesis.amplitude_deltas),
            axis=[1, 2]
        )
        self.loss_fn = self.loss_fn * loss_weight / b.audios["ds_feats"]


class DetNoiseRMSRatioLoss(object):
    def __init__(self, attack_graph, loss_weight=1.0):
        g, b = attack_graph, attack_graph.batch

        def rms(x):
            return tf.sqrt(tf.abs(tf.reduce_mean(tf.square(x) + 1e-8, axis=1)))

        self.loss_fn = tf.abs(rms(g.synthesis.det.deltas) - rms(g.synthesis.noise.deltas))
        self.loss_fn = self.loss_fn * loss_weight


class SpectralLoss(object):
    def __init__(self, attack, norm: int = 2, loss_weight: float = 1e-3):

        self.magnitude_diff = tf.cast(
            tf.abs(attack.graph.synthesiser.stft_deltas), tf.float32
        )

        self.mag_loss_fn = tf.reduce_mean(self.magnitude_diff ** norm, axis=[1, 2])

        self.loss_fn = loss_weight * self.mag_loss_fn


class UpdateOnDecodingSynth(UpdateOnDecoding):

    def decode_step_logic(self):

        # we can't do the rounding for synthesis attack unfortunately.
        # (at least I don't think so? maybe it's just slightly more complex?)

        # deltas = self.attack.sess.run(self.attack.graph.raw_deltas)
        # self.attack.sess.run(
        #     self.attack.graph.raw_deltas.assign(tf.round(deltas))
        # )

        # can use either tf or deepspeech decodings ("ds" or "batch")
        # "batch" is prefered as it's what the actual model would use.
        # It does mean switching to CPU every time we want to do
        # inference but it's not a major hit to performance

        # keep the top 5 scoring decodings and their probabilities as that might
        # be useful come analysis time...

        top_5_decodings, top_5_probs = self.attack.victim.inference(
            self.attack.batch,
            feed=self.attack.feeds.attack,
            decoder="batch",
            top_five=True,
        )

        decodings, probs = self.attack.victim.inference(
            self.attack.batch,
            feed=self.attack.feeds.attack,
            decoder="batch",
            top_five=False,
        )

        targets = self.attack.batch.targets["phrases"]

        return {
            "step": self.current_step,
            "data": [
                {
                    "idx": idx,
                    "success": success,
                    "decodings": decodings[idx],
                    "target_phrase": targets[idx],
                    "probs": probs[idx],
                    "top_five_decodings": top_5_decodings[idx],
                    "top_five_probs": top_5_probs[idx],
                }
                for idx, success in self.check_for_success(decodings, targets)
            ]
        }


class UpdateOnLossSynth(UpdateOnLoss):

    def decode_step_logic(self):

        loss = self.tf_run(self.attack.loss_fn)

        top_5_decodings, top_5_probs = self.attack.victim.inference(
            self.attack.batch,
            feed=self.attack.feeds.attack,
            decoder="batch",
            top_five=True,
        )

        decodings, probs = self.attack.victim.inference(
            self.attack.batch,
            feed=self.attack.feeds.attack,
            decoder="batch",
            top_five=False,
        )

        target_loss = [self.loss_bound for _ in range(self.attack.batch.size)]
        targets = self.attack.batch.targets["phrases"]

        return {
            "step": self.current_step,
            "data": [
                {
                    "idx": idx,
                    "success": success,
                    "decodings": decodings[idx],
                    "target_phrase": targets[idx],
                    "top_five_decodings": top_5_decodings[idx],
                    "top_five_probs": top_5_probs[idx],
                    "probs": probs[idx]
                }
                for idx, success in self.check_for_success(loss, target_loss)
            ]
        }
