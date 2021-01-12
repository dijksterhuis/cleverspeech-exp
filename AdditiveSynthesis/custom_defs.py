import tensorflow as tf
import numpy as np

from cleverspeech.Attacks.Base import Placeholders
from cleverspeech.Utils import np_arr, np_zero, np_one, lcomp


class SynthesisAttack:
    def __init__(self, sess, batch, hard_constraint, synthesis):

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
        max_len = batch.audios.max_length
        act_lengths = batch.audios.actual_lengths

        self.placeholders = Placeholders(batch_size, max_len)

        self.masks = tf.Variable(
            tf.zeros([batch_size, max_len]),
            trainable=False,
            validate_shape=True,
            dtype=tf.float32,
            name='qq_masks'
        )

        # Generate the delta synth parameter objects which we will optimise
        deltas = synthesis.synthesise()

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

        self.opt_vars = synthesis.opt_vars


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

        assert b.audios.feature_lengths.all() > 0

        self.loss_fn = tf.reduce_sum(
            tf.abs(g.synthesis.amplitude_deltas),
            axis=[1, 2]
        )
        self.loss_fn = self.loss_fn * loss_weight / b.audios.feature_lengths


class AdditiveEnergyLoss(object):
    def __init__(self, attack_graph, loss_weight=1.0):

        g, b = attack_graph, attack_graph.batch

        assert b.audios.feature_lengths.all() > 0

        self.loss_fn = tf.reduce_sum(
            tf.square(g.synthesis.amplitude_deltas),
            axis=[1, 2]
        )
        self.loss_fn = self.loss_fn * loss_weight / b.audios.feature_lengths


class DetNoiseRMSRatioLoss(object):
    def __init__(self, attack_graph, loss_weight=1.0):
        g, b = attack_graph, attack_graph.batch

        def rms(x):
            return tf.sqrt(tf.abs(tf.reduce_mean(tf.square(x) + 1e-8, axis=1)))

        self.loss_fn = tf.abs(rms(g.synthesis.det.deltas) - rms(g.synthesis.noise.deltas))
        self.loss_fn = self.loss_fn * loss_weight

