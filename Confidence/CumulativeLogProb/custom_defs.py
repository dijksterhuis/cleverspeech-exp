from collections import OrderedDict

import tensorflow as tf

from cleverspeech.graph.Losses import BaseLogitDiffLoss
from cleverspeech.graph.Optimisers import AdamOptimiser
from cleverspeech.utils.Utils import lcomp
from cleverspeech.graph.Outputs import Base as Outputs


class BaseLogProbsLoss(BaseLogitDiffLoss):
    def __init__(self, attack_graph, target_argmax, weight_settings=(None, None)):

        super().__init__(
            attack_graph,
            target_argmax,
            weight_settings=weight_settings,
            softmax=True
        )

        self.log_smax = tf.log(self.target_logit)

        self.fwd_target = self.target_probs(self.log_smax)
        self.back_target = self.target_probs(self.log_smax, backward_pass=True)

        self.fwd_target_log_probs = self.fwd_target[:, -1]
        self.back_target_log_probs = self.back_target[:, -1]

    @staticmethod
    def target_probs(x_t, backward_pass=False):
        return tf.cumsum(
            x_t,
            exclusive=False,
            reverse=backward_pass,
            axis=1
        )


class FwdOnlyLogProbsLoss(BaseLogProbsLoss):
    def __init__(self, attack_graph, target_argmax, weight_settings=(-1.0, -1.0)):
        """
        """

        super().__init__(
            attack_graph,
            target_argmax,
            weight_settings=weight_settings,
        )

        self.loss_fn = self.fwd_target_log_probs
        self.loss_fn *= self.weights

        self.grads = self.back_target_log_probs


class BackOnlyLogProbsLoss(BaseLogProbsLoss):
    def __init__(self, attack_graph, target_argmax, weight_settings=(-1.0, -1.0)):
        """
        """

        super().__init__(
            attack_graph,
            target_argmax,
            weight_settings=weight_settings,
        )

        self.loss_fn = self.back_target_log_probs
        self.loss_fn *= self.weights


class FwdPlusBackLogProbsLoss(BaseLogProbsLoss):
    def __init__(self, attack_graph, target_argmax, weight_settings=(-1.0, -1.0)):
        """
        """

        super().__init__(
            attack_graph,
            target_argmax,
            weight_settings=weight_settings,
        )

        self.loss_fn = self.fwd_target_log_probs + self.back_target_log_probs
        self.loss_fn *= self.weights


class FwdMultBackLogProbsLoss(BaseLogProbsLoss):
    def __init__(self, attack_graph, target_argmax, weight_settings=(-1.0, -1.0)):
        """
        """

        super().__init__(
            attack_graph,
            target_argmax,
            weight_settings=weight_settings,
        )

        self.loss_fn = self.fwd_target_log_probs * self.back_target_log_probs
        self.loss_fn *= self.weights


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


class LogProbOutputs(Outputs):
    def custom_logging_modifications(self, log_output, batch_idx):

        # Display in log files the current target alignment#s forward log
        # probability and the log probability of the most likely alignment
        # calculated by viberti

        target_log_probs = self.attack.loss[0].fwd_target_log_probs
        most_likely_log_probs = self.attack.loss[0].fwd_other_log_probs

        target_log_probs, most_likely_log_probs = self.attack.procedure.tf_run(
            [target_log_probs, most_likely_log_probs]
        )

        additional = OrderedDict(
            [
                ("t_alpha", target_log_probs[batch_idx]),
                ("ml_alpha", most_likely_log_probs[batch_idx])
            ]
        )

        log_output.update(additional)

        return log_output

    def custom_success_modifications(self, db_output, batch_idx):

        # As above, except write it to disk in the result json file

        target_log_probs = self.attack.loss[0].fwd_target_log_probs
        most_likely_log_probs = self.attack.loss[0].fwd_other_log_probs

        target_log_probs, most_likely_log_probs = self.attack.procedure.tf_run(
            [target_log_probs, most_likely_log_probs]
        )

        additional = OrderedDict(
            [
                ("target_alignment_alpha_log_prob", target_log_probs[batch_idx]),
                ("most_likely_alignment_alpha_log_prob", most_likely_log_probs[batch_idx])
            ]
        )

        db_output.update(additional)

        return db_output

