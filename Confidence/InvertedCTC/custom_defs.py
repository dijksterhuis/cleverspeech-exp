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


class AntiCTC(BaseLoss):
    def __init__(self, attack_graph, alignment=None, weight_settings=(1.0, 1.0)):

        """
        Goal: Make all other logits values *less likely* by optimising with CTC
        loss for all valid alignments *except* our target alignment.
        """

        super().__init__(
            attack_graph.sess,
            attack_graph.batch.size,
            weight_settings=weight_settings
        )

        seq_lengths = attack_graph.batch.audios["ds_feats"]

        self.target_argmax = alignment  # [b x feats]

        # Current logits is [b, feats, chars]
        # current_argmax is for debugging purposes only
        self.current = tf.transpose(attack_graph.victim.raw_logits, [1, 0, 2])

        # Create one hot matrices to multiply by current logits.
        # These essentially act as a filter to keep only the target logit or
        # the rest of the logits (non-target).

        targs_onehot = tf.one_hot(
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

        # AntiCTC sjould make all valid alignments that are *not* the target
        # alignment *less* likely, so we modify the logits to include the
        # classes per frame that are not in our target alignment.

        logits_mod = others_onehot * self.current
        self.targs = tf.reduce_sum(targs_onehot * self.current, axis=2)

        if alignment is not None:
            log("Using CTC alignment search.", wrap=True)
            self.ctc_target = tf.keras.backend.ctc_label_dense_to_sparse(
                alignment,
                attack_graph.batch.audios["ds_feats"],
            )
        else:
            log("Using repeated alignment.", wrap=True)
            self.ctc_target = tf.keras.backend.ctc_label_dense_to_sparse(
                attack_graph.graph.placeholders.targets,
                attack_graph.graph.placeholders.target_lengths,
            )

        logits_shape = attack_graph.victim.raw_logits.get_shape().as_list()

        blank_token_pad = tf.zeros(
            [logits_shape[0], logits_shape[1], 1],
            tf.float32
        )

        self.logits_mod = tf.concat(
            [tf.transpose(logits_mod, [1, 0, 2]), blank_token_pad],
            axis=2
        )

        self.loss_fn = tf.nn.ctc_loss(
            labels=tf.cast(self.ctc_target, tf.int32),
            inputs=self.logits_mod,
            sequence_length=seq_lengths,
            preprocess_collapse_repeated=False,
            ctc_merge_repeated=False,
        ) * self.weights

