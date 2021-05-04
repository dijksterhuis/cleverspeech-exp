import tensorflow as tf

from cleverspeech.graph.Losses import BaseLoss


class OtherTranscriptionCTCLoss(BaseLoss):
    """
    Simple adversarial CTC Loss from https://arxiv.org/abs/1801.01944

    N.B. This loss does *not* conform to l(x + d, t) <= 0 <==> C(x + d) = t
    """
    def __init__(self, attack, weight_settings=(1.0, 1.0)):

        super().__init__(
            attack.sess,
            attack.batch.size,
            weight_settings=weight_settings
        )

        self.ctc_target = tf.keras.backend.ctc_label_dense_to_sparse(
            attack.placeholders.true_targets,
            attack.placeholders.true_target_lengths
        )

        self.loss_fn = tf.nn.ctc_loss(
            labels=tf.cast(self.ctc_target, tf.int32),
            inputs=attack.victim.raw_logits,
            sequence_length=attack.batch.audios["ds_feats"]
        ) * self.weights


class TruePlaceholders(object):

    def __init__(self, batch):

        batch_size, maxlen = batch.size, batch.audios["max_samples"]

        self.audios = tf.placeholder(
            tf.float32, [batch_size, maxlen], name="new_input"
        )
        self.audio_lengths = tf.placeholder(
            tf.int32, [batch_size], name='qq_featlens'
        )
        self.targets = tf.placeholder(
            tf.int32, [batch_size, None], name='qq_targets'
        )
        self.target_lengths = tf.placeholder(
            tf.int32, [batch_size], name='qq_target_lengths'
        )
        self.true_targets = tf.placeholder(
            tf.int32, [batch_size, None], name='qq_true_targets'
        )

        self.true_target_lengths = tf.placeholder(
            tf.int32, [batch_size], name='qq_true_target_lengths'
        )


class TrueFeeds:
    def __init__(self, batch):
        """
        Holds the feeds which will be passed into DeepSpeech for normal or
        attack evaluation.

        :param audio_batch: a batch of audio examples (`Audios` class)
        :param target_batch: a batch of target phrases (`Targets` class)
        """

        self.audio = batch.audios
        self.targets = batch.targets
        self.trues = batch.trues
        self.examples = None
        self.attack = None
        self.alignments = None

    def create_feeds(self, placeholders):
        """
        Create the actual feeds

        :param graph: the attack graph which holds the input placeholders
        :return: feed_dict for both plain examples and attack optimisation
        """
        # TODO - this is nasty!
        self.alignments = {
            placeholders.targets: self.targets["indices"],
            placeholders.target_lengths: self.targets["lengths"],
        }

        self.examples = {
            placeholders.audios: self.audio["padded_audio"],
            placeholders.audio_lengths: self.audio["ds_feats"],
        }

        self.attack = {
            placeholders.audios: self.audio["padded_audio"],
            placeholders.audio_lengths: self.audio["ds_feats"],
            placeholders.targets: self.targets["indices"],
            placeholders.target_lengths: self.targets["lengths"],
            placeholders.true_targets: self.trues["padded_indices"],
            placeholders.true_target_lengths: self.trues["lengths"],
        }

        return self.examples, self.attack

