from collections import OrderedDict

import tensorflow as tf

from cleverspeech.graph.Losses import BaseLogitDiffLoss
from cleverspeech.graph.Optimisers import AdamOptimiser
from cleverspeech.utils.Utils import lcomp


class BaseLogProbsLoss(BaseLogitDiffLoss):
    def __init__(self, attack_graph, target_argmax, weight_settings=(None, None)):

        super().__init__(
            attack_graph,
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
    def __init__(self, attack_graph, target_argmax, weight_settings=(1.0, 1.0)):
        """
        """

        super().__init__(
            attack_graph,
            target_argmax,
            weight_settings=weight_settings,
        )

        self.loss_fn = self.fwd_target_log_probs
        self.loss_fn *= -self.weights


class BackOnlyLogProbsLoss(BaseLogProbsLoss):
    def __init__(self, attack_graph, target_argmax, weight_settings=(1.0, 1.0)):
        """
        """

        super().__init__(
            attack_graph,
            target_argmax,
            weight_settings=weight_settings,
        )

        self.loss_fn = self.back_target_log_probs
        self.loss_fn *= -self.weights


class FwdPlusBackLogProbsLoss(BaseLogProbsLoss):
    def __init__(self, attack_graph, target_argmax, weight_settings=(1.0, 1.0)):
        """
        """

        super().__init__(
            attack_graph,
            target_argmax,
            weight_settings=weight_settings,
        )

        self.loss_fn = self.fwd_target_log_probs + self.back_target_log_probs
        self.loss_fn *= -self.weights


class FwdMultBackLogProbsLoss(BaseLogProbsLoss):
    def __init__(self, attack_graph, target_argmax, weight_settings=(1.0, 1.0)):
        """
        """

        super().__init__(
            attack_graph,
            target_argmax,
            weight_settings=weight_settings,
        )

        self.loss_fn = self.fwd_target_log_probs * self.back_target_log_probs
        self.loss_fn *= -self.weights


class AdamOptimiserWithGrads(AdamOptimiser):
    def create_optimiser(self):

        adv_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
        )

        grad_var = adv_optimizer.compute_gradients(
            self.attack.loss_fn,
            self.attack.graph.opt_vars,
            colocate_gradients_with_ops=True,
            grad_loss=self.attack.loss[0].grads,
        )
        assert None not in lcomp(grad_var, i=0)
        self.train = adv_optimizer.apply_gradients(grad_var)
        self.variables = adv_optimizer.variables()
