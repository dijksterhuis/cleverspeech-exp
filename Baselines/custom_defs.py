import tensorflow as tf

from cleverspeech.graph.Procedures import UpdateOnDecoding
from cleverspeech.utils.Utils import lcomp


class AlignmentLoss(object):
    def __init__(self, alignment_graph):
        feat_lens = alignment_graph.batch.audios["ds_feats"]

        self.ctc_target = tf.keras.backend.ctc_label_dense_to_sparse(
            alignment_graph.graph.targets,
            alignment_graph.graph.target_lengths
        )

        self.loss_fn = tf.nn.ctc_loss(
            labels=self.ctc_target,
            inputs=alignment_graph.graph.raw_alignments,
            sequence_length=feat_lens
        )


class CTCSearchGraph:
    def __init__(self, sess, batch, attack_graph):
        alignment_shape = attack_graph.victim.raw_logits.shape.as_list()

        self.raw_alignments = tf.Variable(
            tf.ones(alignment_shape) * 20,
            dtype=tf.float32,
            trainable=True,
            name='qq_alignment'
        )

        self.logits_alignments = tf.transpose(self.raw_alignments, [1, 0, 2])
        self.softmax_alignments = tf.nn.softmax(self.logits_alignments)
        self.target_alignments = tf.argmax(self.softmax_alignments, axis=2)

        # TODO - this should be loaded from feeds later on
        self.targets = attack_graph.graph.placeholders.targets
        self.target_lengths = attack_graph.graph.placeholders.target_lengths


class CTCAlignmentOptimiser:
    def __init__(self, graph):

        self.graph = graph
        self.loss = self.graph.loss_fn

        self.train_alignment = None
        self.variables = None

    def create_optimiser(self):

        optimizer = tf.train.AdamOptimizer(10)

        grad_var = optimizer.compute_gradients(
            self.graph.loss_fn,
            self.graph.graph.raw_alignments
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
                    g.loss_fn,
                    g.graph.softmax_alignments,
                    g.graph.logits_alignments,
                    self.train_alignment
                ]

            ctc_limit, softmax, raw, _ = g.sess.run(
                train_ops,
                feed_dict=g.feeds.alignments
            )

            decodings, probs = victim.inference(
                b,
                logits=softmax,
                decoder="batch",
                top_five=False
            )

            if all([d == b.targets["phrases"][0] for d in decodings]):
                break


class CTCAlignmentsUpdateHard(UpdateOnDecoding):
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
        opt_vars += [self.alignment_graph.graph.raw_alignments]
        opt_vars += self.attack.optimiser.variables
        opt_vars += self.alignment_graph.optimiser.variables

        self.attack.sess.run(tf.variables_initializer(opt_vars))

    def run(self):

        self.alignment_graph.optimise(self.attack.victim)

        for results in super().run():
            yield results
