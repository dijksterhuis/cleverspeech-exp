import tensorflow as tf
import numpy as np

from cleverspeech.graph.Losses import BaseLoss
from cleverspeech.graph.Procedures import UpdateOnDecoding, UpdateOnLoss
from cleverspeech.utils.Utils import lcomp, log


class AlignmentLoss(object):
    def __init__(self, alignment_graph):
        seq_lens = alignment_graph.batch.audios["real_feats"]

        self.ctc_target = tf.keras.backend.ctc_label_dense_to_sparse(
            alignment_graph.graph.targets,
            alignment_graph.graph.target_lengths
        )

        self.loss_fn = tf.nn.ctc_loss(
            labels=self.ctc_target,
            inputs=alignment_graph.graph.raw_alignments,
            sequence_length=seq_lens,
        )


class CTCSearchGraph:
    def __init__(self, sess, batch, attack_graph):
        batched_alignment_shape = attack_graph.victim.logits.shape.as_list()

        self.initial_alignments = tf.Variable(
            tf.zeros(batched_alignment_shape),
            dtype=tf.float32,
            trainable=True,
            name='qq_alignment'
        )

        # mask is *added* to force decoder to see the logits for those frames as
        # repeat characters. CTC-Loss outputs zero valued vectors for those
        # character classes (as they're beyond the actual alignment length)
        # This messes with decoder output.

        # --> N.B. This is legacy problem with tensorflow/numpy not being able
        # to handle ragged inputs for tf.Variables etc.

        self.mask = tf.Variable(
            tf.ones(batched_alignment_shape),
            dtype=tf.float32,
            trainable=False,
            name='qq_alignment_mask'
        )

        self.logits_alignments = self.initial_alignments + self.mask
        self.raw_alignments = tf.transpose(self.logits_alignments, [1, 0, 2])
        self.softmax_alignments = tf.nn.softmax(self.logits_alignments)
        self.target_alignments = tf.argmax(self.softmax_alignments, axis=2)

        # TODO - this should be loaded from feeds later on
        self.targets = attack_graph.graph.placeholders.targets
        self.target_lengths = attack_graph.graph.placeholders.target_lengths

        per_logit_lengths = batch.audios["real_feats"]
        maxlen = batched_alignment_shape[1]

        initial_masks = np.asarray(
            [m for m in self.gen_mask(per_logit_lengths, maxlen)],
            dtype=np.float32
        )

        sess.run(self.mask.assign(initial_masks))

    @staticmethod
    def gen_mask(per_logit_len, maxlen):
        # per actual frame
        for l in per_logit_len:
            # per possible frame
            masks = []
            for f in range(maxlen):
                if l > f:
                    # if should be optimised
                    mask = np.zeros([29])
                else:
                    # shouldn't be optimised
                    mask = np.zeros([29])
                    mask[28] = 30.0
                masks.append(mask)
            yield np.asarray(masks)


class CTCAlignmentOptimiser:
    def __init__(self, graph):

        self.graph = graph
        self.loss = self.graph.loss_fn

        self.train_alignment = None
        self.variables = None

    def create_optimiser(self):

        optimizer = tf.train.AdamOptimizer(1)

        grad_var = optimizer.compute_gradients(
            self.graph.loss_fn,
            self.graph.graph.initial_alignments
        )
        assert None not in lcomp(grad_var, i=0)

        self.train_alignment = optimizer.apply_gradients(grad_var)
        self.variables = optimizer.variables()

    def optimise(self, batch, victim):

        g, v, b = self.graph, victim, batch

        logits = v.get_logits(v.raw_logits, g.feeds.examples)
        assert logits.shape == g.graph.raw_alignments.shape

        while True:

            train_ops = [
                self.graph.loss_fn,
                g.graph.softmax_alignments,
                g.graph.logits_alignments,
                g.graph.mask,
                self.train_alignment
            ]

            ctc_limit, softmax, raw, m, _ = g.sess.run(
                train_ops,
                feed_dict=g.feeds.alignments
            )

            decodings, probs = victim.inference(
                b,
                logits=softmax,
                decoder="batch",
                top_five=False
            )

            if all([d == b.targets["phrases"][0] for d in decodings]) and all(c < 0.1 for c in ctc_limit):
                s = "Found an alignment for each example:"
                for d, p, t in zip(decodings, probs, b.targets["phrases"]):
                    s += "\nTarget: {t} | Decoding: {d} | Probs: {p:.3f}".format(
                        t=t,
                        d=d,
                        p=p,
                    )
                log(s, wrap=True)
                break


class CTCAlignmentsUpdateOnDecode(UpdateOnDecoding):
    def __init__(self, attack, alignment_graph, *args, **kwargs):
        """
        Initialise the evaluation procedure.

        :param attack_graph: The current attack graph perform optimisation with.
        """

        super().__init__(attack, *args, **kwargs)

        self.alignment_graph = alignment_graph
        self.__init_optimiser_variables()

    def __init_optimiser_variables(self):

        # We must wait until now to initialise the optimiser so that we can
        # initialise only the attack variables (i.e. not the deepspeech ones).

        self.alignment_graph.optimiser.create_optimiser()
        self.attack.optimiser.create_optimiser()

        opt_vars = self.attack.graph.opt_vars
        opt_vars += [self.alignment_graph.graph.initial_alignments]
        opt_vars += self.attack.optimiser.variables
        opt_vars += self.alignment_graph.optimiser.variables

        self.attack.sess.run(tf.variables_initializer(opt_vars))

    def run(self):
        self.alignment_graph.optimise(self.attack.victim)
        for r in super().run():
            yield r


class CTCAlignmentsUpdateOnLoss(UpdateOnLoss):
    def __init__(self, attack, alignment_graph, *args, **kwargs):
        """
        Initialise the evaluation procedure.

        :param attack_graph: The current attack graph perform optimisation with.
        """

        super().__init__(attack, *args, **kwargs)

        self.alignment_graph = alignment_graph
        self.__init_optimiser_variables()

    def __init_optimiser_variables(self):

        # We must wait until now to initialise the optimiser so that we can
        # initialise only the attack variables (i.e. not the deepspeech ones).

        self.alignment_graph.optimiser.create_optimiser()
        self.attack.optimiser.create_optimiser()

        opt_vars = self.attack.graph.opt_vars
        opt_vars += [self.alignment_graph.graph.initial_alignments]
        opt_vars += self.attack.optimiser.variables
        opt_vars += self.alignment_graph.optimiser.variables

        self.attack.sess.run(tf.variables_initializer(opt_vars))

    def run(self):
        self.alignment_graph.optimise(self.attack.victim)
        for r in super().run():
            yield r


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

        exp_framewise_targets = tf.transpose(self.target_logit, [1, 0])
        exp_framewise_others = tf.nn.log_softmax(tf.transpose(self.max_other_logit, [1, 0]))

        max_log_state = tf.reduce_max(exp_framewise_targets, axis=0)

        # the geometric prob product of the target alignment

        self.fwd_target = tf.scan(
            lambda a, x: max_log_state - tf.log(tf.exp(a) * tf.exp(x)),
            exp_framewise_targets
        )
        self.back_target = tf.scan(
            lambda a, x: max_log_state - tf.log(tf.exp(a) * tf.exp(x)),
            exp_framewise_targets,
            reverse=True
        )

        # others only calculates the next likely alignment based on the argmax
        # ==> I need to calculate the [state, state] matrix from CTC loss, but
        # excluding the target alignment. Then take the difference of the target
        # alignment vs. **ALL** other alignments.

        self.fwd_next_likely = tf.scan(
            lambda a, x: max_log_state - tf.log(tf.exp(a) * tf.exp(x)),
            exp_framewise_others
        )
        self.back_next_likely = tf.scan(
            lambda a, x: max_log_state - tf.log(tf.exp(a) * tf.exp(x)),
            exp_framewise_others,
            reverse=True
        )

        # alignment probabilities are the last values in this sequence
        self.max_fwd_target = tf.reduce_max(self.fwd_target, axis=0)
        self.max_fwd_others = tf.reduce_max(self.fwd_next_likely, axis=0)
        self.max_back_target = tf.reduce_max(self.back_target, axis=0)
        self.max_back_others = tf.reduce_max(self.back_next_likely, axis=0)

        self.fwd_prod = -self.fwd_target + self.fwd_next_likely
        self.back_prod = -self.back_target + self.back_next_likely


class FwdPlusBackVibertish(BaseVibertishLoss):
    def __init__(self, attack_graph, target_argmax, weight_settings=(-1.0, -1.0)):
        """
        """

        super().__init__(
            attack_graph,
            target_argmax,
            weight_settings=weight_settings,
        )

        self.prod = self.fwd_prod + self.back_prod
        self.loss_fn = tf.reduce_sum(self.prod, axis=0) * self.weights


class FwdMultBackVibertish(BaseVibertishLoss):
    def __init__(self, attack_graph, target_argmax, weight_settings=(-1.0, -1.0)):
        """
        """

        super().__init__(
            attack_graph,
            target_argmax,
            weight_settings=weight_settings,
        )

        self.prod = -(self.fwd_prod * self.back_prod)
        self.loss_fn = tf.reduce_sum(self.prod, axis=0) * self.weights


class FwdOnlyVibertish(BaseVibertishLoss):
    def __init__(self, attack_graph, target_argmax, weight_settings=(-1.0, -1.0)):
        """
        """

        super().__init__(
            attack_graph,
            target_argmax,
            weight_settings=weight_settings,
        )

        self.prod = self.fwd_prod
        self.loss_fn = tf.reduce_sum(self.prod, axis=0) * self.weights


class BackOnlyVibertish(BaseVibertishLoss):
    def __init__(self, attack_graph, target_argmax, weight_settings=(-1.0, -1.0)):
        """
        """

        super().__init__(
            attack_graph,
            target_argmax,
            weight_settings=weight_settings,
        )

        self.prod = self.back_prod
        self.loss_fn = tf.reduce_sum(self.prod, axis=0) * self.weights


