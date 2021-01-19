import tensorflow as tf


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
    def __init__(self, attack_graph, loss_weight=1.0):

        self.ctc_target = tf.keras.backend.ctc_label_dense_to_sparse(
            attack_graph.graph.placeholders.targets,
            attack_graph.graph.placeholders.target_lengths,
        )

        blank_pad = tf.zeros(
            [
                attack_graph.batch.audios.feature_lengths[0] + 1,
                attack_graph.batch.size,
                1
            ],
            tf.float32
        )

        logits_mod = tf.concat(
            [attack_graph.victim.raw_logits, blank_pad],
            axis=2
        )

        self.loss_fn = tf.nn.ctc_loss(
            labels=tf.cast(self.ctc_target, tf.int32),
            inputs=logits_mod,
            sequence_length=attack_graph.batch.audios.feature_lengths,
            preprocess_collapse_repeated=False,
            ctc_merge_repeated=False,
        ) * loss_weight
