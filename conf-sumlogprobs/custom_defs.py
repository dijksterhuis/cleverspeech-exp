from collections import OrderedDict

import tensorflow as tf

from cleverspeech.graph.Losses import BaseLogitDiffLoss
from cleverspeech.utils.Utils import lcomp


class BaseLogProbsLoss(BaseLogitDiffLoss):
    def __init__(self, attack, target_argmax, weight_settings=(None, None)):

        super().__init__(
            attack,
            target_argmax,
            weight_settings=weight_settings,
            softmax=True
        )

        self.log_smax = tf.log(self.target_logit)

        self.fwd_target_log_probs = tf.reduce_sum(self.log_smax, axis=-1)
        self.back_target_log_probs = tf.reduce_sum(tf.reverse(self.log_smax, axis=[-1]), axis=-1)

    @staticmethod
    def target_probs(x_t, backward_pass=False):
        probability_vector = tf.cumsum(
            x_t,
            exclusive=False,
            reverse=backward_pass,
            axis=1
        )
        if backward_pass:
            probability_vector = tf.reverse(probability_vector, axis=[1])

        return probability_vector


class FwdOnlyLogProbsLoss(BaseLogProbsLoss):
    def __init__(self, attack, target_argmax, weight_settings=(1.0, 1.0)):
        """
        """

        super().__init__(
            attack,
            target_argmax,
            weight_settings=weight_settings,
        )

        self.loss_fn = self.fwd_target_log_probs
        self.loss_fn *= -self.weights


class BackOnlyLogProbsLoss(BaseLogProbsLoss):
    def __init__(self, attack, target_argmax, weight_settings=(1.0, 1.0)):
        """
        """

        super().__init__(
            attack,
            target_argmax,
            weight_settings=weight_settings,
        )

        self.loss_fn = self.back_target_log_probs
        self.loss_fn *= -self.weights


class FwdPlusBackLogProbsLoss(BaseLogProbsLoss):
    def __init__(self, attack, target_argmax, weight_settings=(1.0, 1.0)):
        """
        """

        super().__init__(
            attack,
            target_argmax,
            weight_settings=weight_settings,
        )

        self.loss_fn = self.fwd_target_log_probs + self.back_target_log_probs
        self.loss_fn *= -self.weights


class FwdMultBackLogProbsLoss(BaseLogProbsLoss):
    def __init__(self, attack, target_argmax, weight_settings=(1.0, 1.0)):
        """
        """

        super().__init__(
            attack,
            target_argmax,
            weight_settings=weight_settings,
        )

        self.loss_fn = self.fwd_target_log_probs * self.back_target_log_probs
        self.loss_fn *= -self.weights
