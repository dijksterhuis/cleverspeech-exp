import tensorflow as tf

from cleverspeech.Attacks.Procedures import IterativeHardConstraintUpdate
from cleverspeech.Utils import lcomp


class AlignmentLoss(object):
    def __init__(self, alignment_graph):
        feat_lens = alignment_graph.batch.audios.feature_lengths

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

        print(alignment_shape)

        self.logits_alignments = tf.transpose(self.raw_alignments, [1, 0, 2])
        self.softmax_alignments = tf.nn.softmax(self.logits_alignments)

        # TODO - this should be loaded from feeds later on
        self.targets = attack_graph.graph.placeholders.targets
        self.target_lengths = attack_graph.graph.placeholders.target_lengths

        #super().__init__(attack_graph.sess, batch)


class CTCAlignmentOptimiser:
    def __init__(self, graph):

        self.graph = graph
        self.loss = self.graph.adversarial_loss

        self.train_alignment = None

    def create_optimiser(self):

        optimizer = tf.train.AdamOptimizer(10)

        grad_var = optimizer.compute_gradients(
            self.graph.loss_fn,
            self.graph.graph.raw_alignments
        )
        assert None not in lcomp(grad_var, i=0)

        self.train_alignment = optimizer.apply_gradients(grad_var)

    def optimise(self, batch, victim):

        g, v, b = self.graph, victim, batch

        logits = v.get_logits(v.raw_logits, b.feeds.examples)
        assert logits.shape == g.graph.raw_alignments.shape
        #g.sess.run(g.raw_alignments.assign(logits))

        while True:

            train_ops = [
                    self.loss.loss_fn,
                    g.graph.softmax_alignments,
                    g.graph.logits_alignments,
                    self.train_alignment
                ]

            ctc_limit, softmax, raw, _ = g.sess.run(
                train_ops,
                feed_dict=b.feeds.alignments
            )

            # TODO: Need to test whether this gets the correct decoding output
            #       as i finally got the batch decoder working!
            # TODO: Could we look to MAXIMISE a target alignment? i.e. maximise
            #       the negative log likelihood for an alignment
            # TODO: Could we try to find an adversarial alignment? i.e. where
            #       there are no repeat characters, only characters from the
            #       target transcription
            #       ooooooooopppppppeeeeeeennnnnn vs o------------p------e---n

            #print(np.argmax(softmax, axis=2))

            decodings, probs = victim.inference(
                b,
                logits=softmax,
                decoder="batch",
                top_five=False
            )

            if all([d == b.targets.phrases[0] for d in decodings]):
                break


class CTCAlignmentsUpdateHard(IterativeHardConstraintUpdate):
    def __init__(self, attack, alignment_graph, *args, **kwargs):
        """
        Initialise the evaluation procedure.

        :param attack_graph: The current attack graph perform optimisation with.
        """

        super().__init__(attack, *args, **kwargs)

        self.alignment_graph = alignment_graph

        # We must wait until now to initialise the optimiser so that we can
        # initialise only the attack variables (i.e. not the deepspeech ones).
        start_vars = set(x.name for x in tf.global_variables())

        self.alignment_graph.optimiser.create_optimiser()

        attack.optimiser.create_optimiser()

        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # TODO: New base class that holds vars to be *initialised* rather than opt_vars attributes.
        attack.sess.run(tf.variables_initializer(new_vars + attack.graph.opt_vars + [alignment_graph.graph.raw_alignments]))

    def run(self):

        self.alignment_graph.optimise(self.attack.victim)

        for results in super().run():
            yield results
