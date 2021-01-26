import tensorflow as tf
import numpy as np

from cleverspeech.graph.Procedures import Base
from cleverspeech.utils.Utils import lcomp


class RepeatsCTCLoss(object):
    """
    Adversarial CTC Loss that ignores the blank tokens for higher confidence.
    Does not actually work for adversarial attacks as characters from target
    transcription end up being placed in like `----o--o-oo-o----p- ...` which
    merges down to `ooop ...`

    Only used to demonstrate that regular CTC loss can't do what we want without
    further modification to the back-end (recursive) calculations.

    N.B. This loss does *not* conform to l(x + d, t) <= 0 <==> C(x + d) = t
    """
    def __init__(self, attack_graph, alignment=None, loss_weight=1.0):

        if alignment is not None:
            print("\n\nUsing CTC alignment\n\n")
            self.ctc_target = tf.keras.backend.ctc_label_dense_to_sparse(
                alignment,
                attack_graph.batch.audios.feature_lengths,
            )
        else:
            self.ctc_target = tf.keras.backend.ctc_label_dense_to_sparse(
                attack_graph.graph.placeholders.targets,
                attack_graph.graph.placeholders.target_lengths,
            )

        blank_token_pad = tf.zeros(
            [
                attack_graph.batch.audios.feature_lengths[0],
                attack_graph.batch.size,
                1
            ],
            tf.float32
        )

        logits_mod = tf.concat(
            [attack_graph.victim.raw_logits, blank_token_pad],
            axis=2
        )

        self.loss_fn = tf.nn.ctc_loss(
            labels=tf.cast(self.ctc_target, tf.int32),
            inputs=logits_mod,
            sequence_length=attack_graph.batch.audios.feature_lengths,
            preprocess_collapse_repeated=False,
            ctc_merge_repeated=False,
        ) * loss_weight


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

        # TODO: apply mask

        print(alignment_shape)

        self.logits_alignments = tf.transpose(self.raw_alignments, [1, 0, 2])
        self.softmax_alignments = tf.nn.softmax(self.logits_alignments)
        self.target_alignments = tf.argmax(self.softmax_alignments, axis=2)

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
                    g.graph.target_alignments,
                    self.train_alignment
                ]

            ctc_limit, softmax, raw, targets, _ = g.sess.run(
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
            print(decodings, probs)

            if all([0.1 > c for c in ctc_limit]):
                break
            #if all([d == b.targets.phrases[0] for d in decodings]):
            #    break
            #    #self.target_alignments = np.argmax(softmax,axis=2)


class CTCAlignmentsUpdateHard(Base):
    def __init__(self, attack, alignment_graph, *args, loss_lower_bound=10.0, **kwargs):
        """
        Initialise the evaluation procedure.

        :param attack_graph: The current attack graph perform optimisation with.
        """

        super().__init__(attack, *args, **kwargs)

        self.loss_bound = loss_lower_bound

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
        """
        Do the actual optimisation.
        TODO: Distance metric could be held in AttackGraph or Base?

        :param batch: batch data for the specified model.
        :param steps: total number of steps to run optimisation for.
        :param decode_step: when to check for a successful decoding.
        :return:
            stats for each step of optimisation (loss measurements),
            successful adversarial examples
        """
        attack, g, b = self.attack, self.attack.graph, self.attack.batch

        self.alignment_graph.optimise(self.attack.victim)

        while self.current_step < self.steps:

            self.current_step += 1
            step = self.current_step

            attack.optimiser.optimise(b.feeds.attack)

            if step % self.decode_step == 0:
                # can use either tf or deepspeech decodings
                # we prefer ds as it's what the actual model would use.
                # It does mean switching to CPU every time we want to do
                # inference but there's not a major hit to performance

                decodings, probs = attack.victim.inference(
                    b,
                    feed=b.feeds.attack,
                    decoder="batch",
                    top_five=False,
                )

                outs = {
                    "step": self.current_step,
                    "data": [],
                }

                targets = b.targets.phrases

                loss = self.tf_run(attack.adversarial_loss.loss_fn)
                target_loss = [self.loss_bound for _ in range(b.size)]

                for idx, success in self.update_success(loss, target_loss):

                    out = {
                        "idx": idx,
                        "success": success,
                        "decodings": decodings[idx],
                        "target_phrase": b.targets.phrases[idx],
                        "probs": probs
                    }

                    outs["data"].append(out)

                yield outs

