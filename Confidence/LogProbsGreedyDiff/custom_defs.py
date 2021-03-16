import tensorflow as tf

from cleverspeech.graph.Losses import BaseLoss, BaseLogitDiffLoss


class BaseVibertishDifferenceLoss(BaseLogitDiffLoss):
    def __init__(self, attack_graph, target_argmax, weight_settings=(None, None)):

        super().__init__(
            attack_graph,
            target_argmax,
            weight_settings=weight_settings,
            softmax=True
        )

        log_smax_target = tf.log(self.target_logit)
        fwd_target = self.target_probs(log_smax_target)
        back_target = self.target_probs(log_smax_target, backward_pass=True)

        # viberti basically does greedy search anyway... so we may as well use
        # the argmax of the current softmax
        log_smax_current = tf.log(tf.reduce_max(self.current, axis=-1))
        fwd_current = self.target_probs(log_smax_current)
        back_current = self.target_probs(log_smax_current, backward_pass=True)

        # comparison log probabilities to calcalute loss.
        self.fwd_target_log_probs = fwd_target[:, -1]
        self.back_target_log_probs = back_target[:, -1]
        self.fwd_current_log_probs = fwd_current[:, -1]
        self.back_current_log_probs = back_current[:, -1]

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


class VibertiMostLikely(BaseLoss):
    def __init__(self, attack_graph, weight_settings=(None, None)):

        super().__init__(
            attack_graph.sess,
            attack_graph.batch.size,
            weight_settings=weight_settings
        )

        smax_log = tf.log(attack_graph.victm.logits + 1e-8)
        self.fwd_most_likely = self.viberti_probs(smax_log)

    def viberti_probs(self, log_softmax, backward_pass=False):
        """
        Calculate the most likely alignment based on the Viberti forward pass

        :param log_softmax: negative value tensor [batch size, n_frames, tokens]
        :return: vector of alpha log probability estimates [n_frames]
        """

        if backward_pass:
            log_softmax = tf.reverse(log_softmax, axis=[1])

        log_probs = self.__viberti(log_softmax)

        return log_probs

    @staticmethod
    def __viberti(log_softmax):
        """
        Run the viberti algorithm on the log softmax outputs to find the log
        probs, per frame, of the most likely alignment.

        TODO: Handle the batch dimension.

        :param frames: unstacked list of tensors of size [batch, tokens]
        :return: vector of cumulative log probabilities [n_frames]
        """
        batch_size = log_softmax.get_shape().as_list()[0]
        res = list()
        best_alpha = tf.zeros([batch_size, 1], tf.float32)
        frames = tf.unstack(log_softmax, axis=1)

        for current_log_probs in frames:

            alpha_log_probs = current_log_probs + best_alpha
            best_alpha = tf.reduce_max(alpha_log_probs, axis=-1)

            res.append(best_alpha)

        return tf.stack(res)


class FwdOnlyVibertish(BaseVibertishDifferenceLoss):
    def __init__(self, attack_graph, target_argmax, kappa=1.1, weight_settings=(1.0, 1.0)):
        """
        """

        assert kappa > 1.0

        super().__init__(
            attack_graph,
            target_argmax,
            weight_settings=weight_settings,
        )

        target = kappa * self.fwd_target_log_probs
        current = self.fwd_current_log_probs

        self.loss_fn = current - target
        self.loss_fn *= self.weights


class BackOnlyVibertish(BaseVibertishDifferenceLoss):
    def __init__(self, attack_graph, target_argmax, kappa=1.1, weight_settings=(1.0, 1.0)):
        """
        """
        assert kappa > 1.0

        super().__init__(
            attack_graph,
            target_argmax,
            weight_settings=weight_settings,
        )

        target = kappa * self.back_target_log_probs
        current = self.back_current_log_probs

        self.loss_fn = current - target
        self.loss_fn *= self.weights


class FwdPlusBackVibertish(BaseVibertishDifferenceLoss):
    def __init__(self, attack_graph, target_argmax, kappa=1.1, weight_settings=(1.0, 1.0)):
        """
        """
        assert kappa > 1.0

        super().__init__(
            attack_graph,
            target_argmax,
            weight_settings=weight_settings,
        )

        target = kappa * self.fwd_target_log_probs + self.back_target_log_probs
        current = self.fwd_current_log_probs + self.back_current_log_probs

        self.loss_fn = current - target
        self.loss_fn *= self.weights


class FwdMultBackVibertish(BaseVibertishDifferenceLoss):
    def __init__(self, attack_graph, target_argmax, kappa=1.1, weight_settings=(1.0, 1.0)):
        """
        """
        assert kappa > 1.0

        super().__init__(
            attack_graph,
            target_argmax,
            weight_settings=weight_settings,
        )

        target = kappa * self.fwd_target_log_probs * self.back_target_log_probs
        current = self.fwd_current_log_probs * self.back_current_log_probs

        self.loss_fn = target - current
        self.loss_fn *= self.weights





