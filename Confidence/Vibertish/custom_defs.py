import tensorflow as tf

from cleverspeech.graph.Losses import BaseLoss


class BaseLogitDiffLoss(BaseLoss):
    def __init__(self, attack_graph, target_argmax, weight_settings=(None, None)):
        """
        This is a modified version of f_{6} from https://arxiv.org/abs/1608.04644
        using the gradient clipping update method.

        Difference of:
        - target logits value (B)
        - max other logits value (A -- 2nd most likely)

        Once  B > A, then B is most likely and we can stop optimising.

        Unless -k > B, then k acts as a confidence threshold and continues
        optimisation.

        This will push B to become even less likely.
        """

        super().__init__(
            attack_graph.sess,
            attack_graph.batch.size,
            weight_settings=weight_settings
        )

        g = attack_graph

        # We only use the argmax of the generated alignments so we don't have
        # to worry about finding "exact" alignments
        # target_logits should be [b, feats, chars]
        self.target_argmax = target_argmax  # [b x feats]

        # Current logits is [b, feats, chars]
        # current_argmax is for debugging purposes only
        self.current = tf.transpose(g.victim.raw_logits, [1, 0, 2])
        self.current = g.victim.logits

        # Create one hot matrices to multiply by current logits.
        # These essentially act as a filter to keep only the target logit or
        # the rest of the logits (non-target).

        targ_onehot = tf.one_hot(
            self.target_argmax,
            self.current.shape.as_list()[2],
            on_value=1.0,
            off_value=0.0
        )

        others_onehot = tf.one_hot(
            self.target_argmax,
            self.current.shape.as_list()[2],
            on_value=0.0,
            off_value=1.0
        )

        self.others = self.current * others_onehot
        self.targ = self.current * targ_onehot

        # Get the maximums of:
        # - target logit (should just be the target logit value)
        # - all other logits (should be next most likely class)

        self.target_logit = tf.reduce_sum(self.targ, axis=2)
        self.max_other_logit = tf.reduce_max(self.others, axis=2)


class BaseVibertishLoss(BaseLogitDiffLoss):
    def __init__(self, attack_graph, target_argmax, weight_settings=(None, None)):

        super().__init__(
            attack_graph,
            target_argmax,
            weight_settings=weight_settings
        )

        n_feats = self.target_logit.get_shape().as_list()[1]

        feats_weight = tf.cast(tf.range(1, n_feats + 1), dtype=tf.float32)
        back_feats_weight = tf.cast(tf.range(n_feats + 1, 1, -1), dtype=tf.float32)

        if (n_feats) // 2 == 0:
            sign = 1
        else:
            sign = -1

        exp_framewise_targets = tf.nn.softmax(tf.transpose(self.target_logit, [1, 0]))
        exp_framewise_others = tf.nn.log_softmax(tf.transpose(self.max_other_logit, [1, 0]))

        max_log_state = tf.reduce_max(exp_framewise_targets, axis=0)

        # the geometric prob product of the target alignment

        # self.fwd_target = tf.scan(
        #     lambda a, x: tf.log(tf.exp(a) * tf.exp(x)) / tf.reduce_sum(a),
        #     exp_framewise_targets
        # )

        # self.back_target = tf.scan(
        #     lambda a, x: max_log_state - tf.log(tf.exp(a) * tf.exp(x)),
        #     exp_framewise_targets,
        #     reverse=True
        # )

        self.log_smax = tf.nn.log_softmax(self.target_logit),

        self.fwd_target = tf.abs(tf.cumprod(
            tf.nn.log_softmax(self.target_logit),
            exclusive=False,
            reverse=False,
            axis=1
        )) / tf.exp(feats_weight)

        self.back_target = tf.abs(tf.cumprod(
            tf.nn.log_softmax(self.target_logit),
            exclusive=False,
            reverse=True,
            axis=1
        )) / tf.exp(back_feats_weight)

        # self.fwd_target_mod = tf.scan(
        #     lambda a, x: a * tf.abs(x),
        #     tf.transpose(tf.nn.log_softmax(self.target_logit), [1, 0])
        # )
        #
        # self.fwd_target_mod = tf.transpose(
        #     self.fwd_target_mod, [1, 0]
        # ) * (self.target_logit)
        #
        # self.back_target_mod = tf.cumprod(
        #     tf.nn.softmax(self.target_logit),
        #     exclusive=False,
        #     reverse=True,
        #     axis=1
        # )

        # others only calculates the next likely alignment based on the argmax
        # ==> I need to calculate the [state, state] matrix from CTC loss, but
        # excluding the target alignment. Then take the difference of the target
        # alignment vs. **ALL** other alignments.

        self.fwd_next_likely = tf.abs(tf.cumprod(
            tf.nn.log_softmax(self.max_other_logit),
            exclusive=False,
            reverse=False,
            axis=1
        )) / tf.exp(feats_weight)

        self.back_next_likely = tf.abs(tf.cumprod(
            tf.nn.log_softmax(self.max_other_logit),
            exclusive=False,
            reverse=True,
            axis=1
        )) / tf.exp(back_feats_weight)


        # self.fwd_next_likely = tf.scan(
        #     lambda a, x: max_log_state - tf.log(tf.exp(a) * tf.exp(x)),
        #     exp_framewise_others
        # )
        # self.back_next_likely = tf.scan(
        #     lambda a, x: max_log_state - tf.log(tf.exp(a) * tf.exp(x)),
        #     exp_framewise_others,
        #     reverse=True
        # )

        # alignment probabilities are the last values in this sequence
        # self.max_fwd_target = tf.reduce_max(self.fwd_target, axis=1)
        self.max_fwd_target = tf.reduce_sum(self.fwd_target, axis=1)
        self.max_fwd_others = tf.reduce_sum(self.fwd_next_likely, axis=1)
        self.max_back_target = tf.reduce_sum(self.back_target, axis=1)
        self.max_back_others = tf.reduce_sum(self.back_next_likely, axis=0)


class FwdOnlyVibertish(BaseVibertishLoss):
    def __init__(self, attack_graph, target_argmax, weight_settings=(1.0, 1.0)):
        """
        """

        super().__init__(
            attack_graph,
            target_argmax,
            weight_settings=weight_settings,
        )

        self.loss_fn = self.max_fwd_target
        self.loss_fn *= self.weights


class BackOnlyVibertish(BaseVibertishLoss):
    def __init__(self, attack_graph, target_argmax, weight_settings=(1.0, 1.0)):
        """
        """

        super().__init__(
            attack_graph,
            target_argmax,
            weight_settings=weight_settings,
        )

        self.loss_fn = self.max_back_target
        self.loss_fn *= self.weights


class FwdPlusBackVibertish(BaseVibertishLoss):
    def __init__(self, attack_graph, target_argmax, weight_settings=(1.0, 1.0)):
        """
        """

        super().__init__(
            attack_graph,
            target_argmax,
            weight_settings=weight_settings,
        )

        self.loss_fn = self.max_fwd_target # + self.max_fwd_others
        self.loss_fn += self.max_back_target # + self.max_back_others
        self.loss_fn *= self.weights


class FwdMultBackVibertish(BaseVibertishLoss):
    def __init__(self, attack_graph, target_argmax, weight_settings=(1.0, 1.0)):
        """
        """

        super().__init__(
            attack_graph,
            target_argmax,
            weight_settings=weight_settings,
        )

        self.loss_fn = self.max_fwd_target * self.max_back_target
        self.loss_fn *= self.weights


