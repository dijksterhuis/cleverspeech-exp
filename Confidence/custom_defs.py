import tensorflow as tf
import numpy as np

from cleverspeech.graph.Procedures import UpdateOnDecoding, UpdateOnLoss
from cleverspeech.graph.Placeholders import Placeholders
from cleverspeech.utils.Utils import np_arr, np_zero, np_one, lcomp, log


class HiScoresFeed:
    def __init__(self, audio_batch, target_batch):
        """
        Holds the feeds which will be passed into DeepSpeech for normal or
        attack evaluation.

        :param audio_batch: a batch of audio examples (`Audios` class)
        :param target_batch: a batch of target phrases (`Targets` class)
        """

        self.audio = audio_batch
        self.targets = target_batch
        self.examples = None
        self.attack = None
        self.alignments = None

    def create_feeds(self, graph):
        """
        Create the actual feeds

        :param graph: the attack graph which holds the input placeholders
        :return: feed_dict for both plain examples and attack optimisation
        """
        # TODO - this is nasty!

        self.examples = {
            graph.placeholders.audios: self.audio.padded_audio,
            graph.placeholders.audio_lengths: self.audio.feature_lengths
        }

        # TODO new logits placeholder.
        self.attack = {
            graph.placeholders.audios: self.audio.padded_audio,
            graph.placeholders.audio_lengths: self.audio.feature_lengths,
            graph.placeholders.target_logits: self.targets.logits,
            graph.placeholders.logit_indices: self.targets.alignment_indices,
            graph.placeholders.kappas: self.targets.kappa
        }

        return self.examples, self.attack


class HiScoresAttack:
    def __init__(self, sess, batch, hard_constraint):

        """
        Attack graph from 2018 Carlini & Wager Targeted Audio Attack.
        # TODO: norm should be a class defined in Audio.Distance
        # TODO: Distance classes should have `bound()` and `analyse()` methods

        :param sess: tf.Session which will be used in the attack.
        :param tau: the upper bound for the size of the perturbation
        :param batch: The batch of fata
        :param synthesis: the `Audio.Synthesis` class which generates the perturbation
        :param constraint: the `Audio.Distance` class being used for the attack

        """
        batch_size = batch.size
        max_len = batch.audios.max_length
        act_lengths = batch.audios.actual_lengths

        self.placeholders = Placeholders(batch_size, max_len)
        self.placeholders.target_logits = tf.placeholder(
            tf.float32,
            [
                batch_size,
                batch.targets.maximum_length,
                len(batch.targets.tokens)
            ],
            name='qq_target_logits'
        )

        self.placeholders.logit_indices = tf.placeholder(
            tf.int32,
            [
                batch_size,
                batch.targets.maximum_length,
            ],
            name='qq_target_argmaxes'
        )
        self.placeholders.kappas = tf.placeholder(
            tf.float32,
            [
                batch_size,
                1
            ],
            name='qq_target_kappa'
        )

        self.masks = tf.Variable(
            tf.zeros([batch_size, max_len]),
            trainable=False,
            validate_shape=True,
            dtype=tf.float32,
            name='qq_masks'
        )

        # Generate the delta synth parameter objects which we will optimise
        raw_deltas = tf.Variable(
            tf.zeros([batch.size, batch.audios.max_length], dtype=tf.float32),
            trainable=True,
            validate_shape=True,
            dtype=tf.float32,
            name='qq_delta'
        )

        # Mask deltas first so we zero value *any part of the signal* that is
        # zero value padded in the original audio
        deltas = raw_deltas * self.masks

        # Restrict delta to valid space before applying constraints

        lower = -2.0 ** 15
        upper = 2.0 ** 15 - 1

        valid_deltas = tf.clip_by_value(
            deltas,
            clip_value_min=lower,
            clip_value_max=upper
        )

        self.final_deltas = hard_constraint.clip(valid_deltas)

        # clip example to valid range
        self.adversarial_examples = tf.clip_by_value(
            self.final_deltas + self.placeholders.audios,
            clip_value_min=lower,
            clip_value_max=upper
        )

        # logits

        self.logits_mask = tf.Variable(
            tf.zeros([batch_size, batch.targets.maximum_length]),
            trainable=False,
            validate_shape=True,
            dtype=tf.float32,
            name='qq_logits_masks'
        )

        self.target_logits = self.logits_mask[:, :, tf.newaxis] * self.placeholders.target_logits

        # initialise static variables
        initial_masks = np_arr(
            lcomp(self._gen_mask(act_lengths, max_len)),
            np.float32
        )

        initial_logits_masks = np_arr(
            lcomp(self._gen_mask(batch.targets.lengths, batch.targets.maximum_length)),
            np.float32
        )

        sess.run(
            (
                self.masks.assign(initial_masks),
                self.logits_mask.assign(initial_logits_masks),
            )
        )

        self.opt_vars = [raw_deltas]


    @staticmethod
    def _gen_mask(lengths, max_len):
        """
        Generate the zero / one valued mask for the perturbations in a batch
        Mask value == 1 when actual < max, == 0 when actual > max

        :param lengths: number of samples per audio example in a batch
        :param max_len: maximum numb. of samples of audio examples in a batch
        :yield: 0/1 valued mask vector of length max_len
        """
        for l in lengths:
            m = list()
            for i in range(max_len):
                if i < l:
                    m.append(1)
                else:
                    m.append(0)
            yield m


class HiScoresCWMaxDiff(object):
    def __init__(self, attack_graph, target_logits, k=0.5, loss_weight=1.0):
        """
        This is f_{6} from https://arxiv.org/abs/1608.04644 using the gradient
        clipping update method.

        Difference of:
        - target logits value (B)
        - max other logits value (A -- 2nd most likely)

        Once  B > A, then B is most likely and we can stop optimising.

        Unless -k > B, then k acts as a confidence threshold and continues
        optimisation.

        This will push B to become even less likely.

        N.B. This loss does *not* seems to conform to:
            l(x + d, t) <= 0 <==> C(x + d) = t

        But it does a much better job than the ArgmaxLowConfidence
        implementation as 0 <= l(x + d, t) < 1.0 for a successful decoding

        TODO: This needs testing.
        TODO: normalise to 0 <= x + d <= 1 and convert to tanh space for `change
              of variable` optimisation
        """

        # We have to set k > 0 for this loss function because k = 0 will only
        # cause the probability of the target character to exactly match the
        # next most likely character...
        # Which means we wouldn't ever achieve success!
        assert type(k) is float
        assert k > 0.0

        g = attack_graph

        # We only use the argmax of the generated alignments so we don't have
        # to worry about finding "exact" alignments
        # target_logits should be [b, feats, chars]
        self.target = target_logits
        self.target_argmax = tf.argmax(self.target, dimension=2)  # [b x feats]

        # Current logits is [b, feats, chars]
        # current_argmax is for debugging purposes only
        self.current = tf.transpose(g.victim.raw_logits, [1, 0, 2])
        self.current_argmax = tf.argmax(self.current, axis=2)

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

        # + 1e-10 covers logit == zero edge case in subsequent where clauses
        # self.others = (self.current + 1e-6) * others_onehot
        # self.targ = (self.current + 1e-6) * targ_onehot

        # DeepSpeech zero-valued Softmax logits are anything < 0.
        # If we don't replace the zero off + on values after one hot
        # multiplication then optimisation could be halted prematurely => the
        # zero can become the most likely class in some cases

        # self.targ = tf.where(
        #     tf.equal(self.targ, tf.zeros(self.targ.shape, dtype=tf.float32)),
        #     -40.0 * tf.ones(self.targ.shape, dtype=tf.float32),
        #     self.targ
        # )
        #
        # self.others = tf.where(
        #     tf.equal(self.others, tf.zeros(self.others.shape, dtype=tf.float32)),
        #     -40.0 * tf.ones(self.others.shape, dtype=tf.float32),
        #     self.others
        # )

        # Get the maximums of:
        # - target logit (should just be the target logit value)
        # - all other logits (should be next most likely class)

        self.target_logit = tf.reduce_max(self.targ, axis=2)
        self.max_other_logit = tf.reduce_max(self.others, axis=2)

        # If target logit is most likely, then the optimiser has done a good job
        # and loss will become negative.
        # Keep optimising until we reached the confidence threshold -- how much
        # distance between logits do we want can be controlled by k

        self.max_diff_abs = self.max_other_logit - self.target_logit

        # MR addition
        # Multiply by minimisation weighting only when target < next class to
        # encourage further optimisation -- we want *highly confident* examples

        # self.max_diff = tf.where(
        #     tf.less_equal(self.max_diff, tf.zeros(self.max_diff.shape, dtype=tf.float32)),
        #     tf.maximum(self.max_diff, -k),
        #     self.max_diff * importance
        # )
        # TODO: per character importance -- could this be adaptive?
        #       e.g. use percentage maxdiff compared to abs. max of maxdiff for
        #       each character? Then, when a character is *VERY* different, it
        #       gets a higher weighting, while other get a lower weighting
        #       (i.e. the characters which are already closer).

        # hacky implementation of character importance, only dealing with
        # repeats and space characters
        #
        # self.max_diff_character_weighted = tf.where(
        #     tf.equal(self.target_argmax, 28),
        #     0.5 * self.max_diff_abs,
        #     self.max_diff_abs,
        # )
        #
        # self.max_diff_character_weighted = tf.where(
        #     tf.equal(self.target_argmax, 0),
        #     0.5 * self.max_diff_abs,
        #     self.max_diff_abs,
        # )
        # Take the maximum between the max diffs or confidence threshold.
        # We add k at the end so the loss is always non-negative.
        # This is for sanity checks and has no impact on optimisation.

        self.max_diff = tf.maximum(self.max_diff_abs, -k) + k
        self.loss_fn = tf.reduce_sum(self.max_diff, axis=1)

        self.loss_fn = self.loss_fn * loss_weight


class HiScoresSquaredMaxDiffLoss(object):
    """
    Literally just calculate the error between current encoder output and
    target logits. The HiScores target logits fuzzing handles most of the
    legwork prior to running any optimisation.

    Note -- the *only* per-character difference should be the class we want to
    change to. This counters a possible issue with the MaxDiff (L2-CW) loss
    where the attack might try to make all other characters less likely compared
    to the target character.
    """
    def __init__(self, attack_graph, loss_weight=1.0, k=0):

        g = attack_graph

        # target_logits should be [b, feats], argmax should be [b, feats]
        self.target = attack_graph.graph.target_logits
        self.target_argmax = attack_graph.graph.placeholders.argmaxes

        # Current logits should be [b, feats, chars]
        self.current = tf.transpose(g.victim.raw_logits, [1, 0, 2])

        # Create one hot matrices to multiply by current logits.
        # These essentially act as a filter to keep only the logit we care about
        # for each feature frame

        onehot = tf.one_hot(
            self.target_argmax,
            self.current.shape.as_list()[2],
            on_value=1.0,
            off_value=0.0
        )
        # filter out the other logits
        self.others = self.current * onehot

        # reduce the tensor to [b, feats, 1]
        self.max_other_logit = tf.reduce_sum(self.others, axis=2)

        self.diff = self.target - self.max_other_logit

        # kappa is not necessarily > 0 with a beam search decoder => the
        # difference between original + target logits can be negative as the
        # bs decoder cares about the likelihood over *all alignment path steps*.

        # That's is why we repeat the characters when doing searches...

        # also, kappa == 0 will return the same transcription as the original,
        # so the loss would be zero

        # so we take the sign of the kappa value from the logits search and
        # invert the difference to account for each case:

        # k > 0 :=> +1 * (target - current) => target - current
        # k < 0 :=> -1 * (target - current) => current - target
        # k == 0 :=> 0 * (target - current) => 0

        # then we take the maximum between each logit and 0, as per
        # Carlini & Wagner's improved loss function

        signs = tf.sign(attack_graph.graph.placeholders.kappas)

        self.complete_diff = tf.maximum(signs * self.diff, 0)

        # penalise characters that are further away than others without causing
        # character differences <0 to become smaller as a result
        self.complete_diff = tf.square(self.complete_diff + 1) - 1

        self.loss_fn = tf.reduce_sum(self.complete_diff, axis=1) * loss_weight


class HiScoresSquaredDiffLoss(object):
    """
    Literally just calculate the error between current encoder output and
    target logits. The HiScores target logits fuzzing handles most of the
    legwork prior to running any optimisation.

    Note -- the *only* per-character difference should be the class we want to
    change to. This counters a possible issue with the MaxDiff (L2-CW) loss
    where the attack might try to make all other characters less likely compared
    to the target character.
    """
    def __init__(self, attack_graph, loss_weight=1.0, k=0):

        g = attack_graph

        # target_logits should be [b, feats], argmax should be [b, feats]
        self.target = attack_graph.graph.target_logits
        self.target_argmax = attack_graph.graph.placeholders.argmaxes

        # Current logits should be [b, feats, chars]
        self.current = tf.transpose(g.victim.raw_logits, [1, 0, 2])

        # Create one hot matrices to multiply by current logits.
        # These essentially act as a filter to keep only the logit we care about
        # for each feature frame

        onehot = tf.one_hot(
            self.target_argmax,
            self.current.shape.as_list()[2],
            on_value=1.0,
            off_value=0.0
        )
        # filter out the other logits
        self.others = self.current * onehot

        # reduce the tensor to [b, feats, 1]
        self.max_other_logit = tf.reduce_sum(self.others, axis=2)

        self.diff = self.target - self.max_other_logit

        # kappa is not necessarily > 0 with a beam search decoder => the
        # difference between original + target logits can be negative as the
        # bs decoder cares about the likelihood over *all alignment path steps*.

        # That's is why we repeat the characters when doing searches...

        # also, kappa == 0 will return the same transcription as the original,
        # so the loss would be zero

        # so we take the sign of the kappa value from the logits search and
        # invert the difference to account for each case:

        # k > 0 :=> +1 * (target - current) => target - current
        # k < 0 :=> -1 * (target - current) => current - target
        # k == 0 :=> 0 * (target - current) => 0

        # then we take the maximum between each logit and 0, as per
        # Carlini & Wagner's improved loss function

        signs = tf.sign(attack_graph.graph.placeholders.kappas)

        # penalise characters that are further away than others without causing
        # character differences <0 to become smaller as a result
        self.complete_diff = tf.square(signs * self.diff + 1) - 1

        self.loss_fn = tf.reduce_sum(self.complete_diff, axis=1) * loss_weight


class HiScoresAbsoluteDiffLoss(object):
    """
    Literally just calculate the error between current encoder output and
    target logits. The HiScores target logits fuzzing handles most of the
    legwork prior to running any optimisation.

    Note -- the *only* per-character difference should be the class we want to
    change to. This counters a possible issue with the MaxDiff (L2-CW) loss
    where the attack might try to make all other characters less likely compared
    to the target character.
    """
    def __init__(self, attack_graph, loss_weight=1.0, k=0):

        g = attack_graph

        # target_logits should be [b, feats], argmax should be [b, feats]
        self.target = attack_graph.graph.target_logits
        self.target_argmax = attack_graph.graph.placeholders.logit_indices

        # Current logits should be [b, feats, chars]
        self.current = tf.transpose(g.victim.raw_logits, [1, 0, 2])

        # Create one hot matrices to multiply by current logits.
        # These essentially act as a filter to keep only the logit we care about
        # for each feature frame

        onehot = tf.one_hot(
            self.target_argmax,
            self.current.shape.as_list()[2],
            on_value=1.0,
            off_value=0.0
        )
        # filter out the other logits
        self.others = self.current * onehot

        # reduce the tensor to [b, feats, 1]
        self.max_other_logit = tf.reduce_sum(self.others, axis=2)

        self.diff = self.target - self.max_other_logit

        # kappa is not necessarily > 0 with a beam search decoder => the
        # difference between original + target logits can be negative as the
        # bs decoder cares about the likelihood over *all alignment path steps*.

        # That's is why we repeat the characters when doing searches...

        # also, kappa == 0 will return the same transcription as the original,
        # so the loss would be zero

        # so we take the sign of the kappa value from the logits search and
        # invert the difference to account for each case:

        # k > 0 :=> +1 * (target - current) => target - current
        # k < 0 :=> -1 * (target - current) => current - target
        # k == 0 :=> 0 * (target - current) => 0

        # then we take the maximum between each logit and 0, as per
        # Carlini & Wagner's improved loss function

        signs = tf.sign(attack_graph.graph.placeholders.kappas)

        # penalise characters that are further away than others without causing
        # character differences <0 to become smaller as a result
        self.complete_diff = tf.abs(signs * self.diff)

        self.loss_fn = tf.reduce_sum(self.complete_diff, axis=1) * loss_weight


class HiScoresSquareAbsDiffLoss(object):
    """
    Literally just calculate the error between current encoder output and
    target logits. The HiScores target logits fuzzing handles most of the
    legwork prior to running any optimisation.

    Note -- the *only* per-character difference should be the class we want to
    change to. This counters a possible issue with the MaxDiff (L2-CW) loss
    where the attack might try to make all other characters less likely compared
    to the target character.
    """
    def __init__(self, attack_graph, loss_weight=1.0, k=0):

        g = attack_graph

        # target_logits should be [b, feats], argmax should be [b, feats]
        self.target = attack_graph.graph.target_logits
        self.target_argmax = attack_graph.graph.placeholders.logit_indices

        # Current logits should be [b, feats, chars]
        self.current = tf.transpose(g.victim.raw_logits, [1, 0, 2])

        # Create one hot matrices to multiply by current logits.
        # These essentially act as a filter to keep only the logit we care about
        # for each feature frame

        onehot = tf.one_hot(
            self.target_argmax,
            self.current.shape.as_list()[2],
            on_value=1.0,
            off_value=0.0
        )
        # filter out the other logits
        self.others = self.current * onehot

        # reduce the tensor to [b, feats, 1]
        self.max_other_logit = tf.reduce_sum(self.others, axis=2)

        self.diff = self.target - self.max_other_logit

        # kappa is not necessarily > 0 with a beam search decoder => the
        # difference between original + target logits can be negative as the
        # bs decoder cares about the likelihood over *all alignment path steps*.

        # That's is why we repeat the characters when doing searches...

        # also, kappa == 0 will return the same transcription as the original,
        # so the loss would be zero

        # so we take the sign of the kappa value from the logits search and
        # invert the difference to account for each case:

        # k > 0 :=> +1 * (target - current) => target - current
        # k < 0 :=> -1 * (target - current) => current - target
        # k == 0 :=> 0 * (target - current) => 0

        # then we take the maximum between each logit and 0, as per
        # Carlini & Wagner's improved loss function

        signs = tf.sign(attack_graph.graph.placeholders.kappas)

        # penalise characters that are further away than others without causing
        # character differences <0 to become smaller as a result
        self.complete_diff = tf.square(tf.abs(signs * self.diff))

        self.loss_fn = tf.reduce_sum(self.complete_diff, axis=1) * loss_weight


class HiScoresLogSumExpLoss(object):
    """
    Literally just calculate the error between current encoder output and
    target logits. The HiScores target logits fuzzing handles most of the
    legwork prior to running any optimisation.

    Note -- the *only* per-character difference should be the class we want to
    change to. This counters a possible issue with the MaxDiff (L2-CW) loss
    where the attack might try to make all other characters less likely compared
    to the target character.
    """
    def __init__(self, attack_graph, loss_weight=1.0, k=0):

        g = attack_graph

        # target_logits should be [b, feats], argmax should be [b, feats]
        self.target = attack_graph.graph.target_logits

        # Current logits should be [b, feats, chars]
        self.current = tf.transpose(g.victim.raw_logits, [1, 0, 2])

        # Create one hot matrices to multiply by current logits.
        # These essentially act as a filter to keep only the logit we care about
        # for each feature frame

        # reduce the tensor to [b, feats, 1]
        self.logsumexp_c = tf.log(tf.reduce_sum(tf.exp(self.current), axis=2))
        self.logsumexp_t = tf.log(tf.reduce_sum(tf.exp(self.target), axis=2))

        self.diff = self.logsumexp_t - self.logsumexp_c

        # kappa is not necessarily > 0 with a beam search decoder => the
        # difference between original + target logits can be negative as the
        # bs decoder cares about the likelihood over *all alignment path steps*.

        # That's is why we repeat the characters when doing searches...

        # also, kappa == 0 will return the same transcription as the original,
        # so the loss would be zero

        # so we take the sign of the kappa value from the logits search and
        # invert the difference to account for each case:

        # k > 0 :=> +1 * (target - current) => target - current
        # k < 0 :=> -1 * (target - current) => current - target
        # k == 0 :=> 0 * (target - current) => 0

        # then we take the maximum between each logit and 0, as per
        # Carlini & Wagner's improved loss function

        signs = tf.sign(attack_graph.graph.placeholders.kappas)

        # penalise characters that are further away than others without causing
        # character differences <0 to become smaller as a result
        self.complete_diff = tf.abs(self.diff)

        self.loss_fn = tf.reduce_sum(self.complete_diff, axis=1) * loss_weight


class HiScoresAbsLoss(object):
    """
    Literally just calculate the error between current encoder output and
    target logits. The HiScores target logits fuzzing handles most of the
    legwork prior to running any optimisation.

    Note -- the *only* per-character difference should be the class we want to
    change to. This counters a possible issue with the MaxDiff (L2-CW) loss
    where the attack might try to make all other characters less likely compared
    to the target character.
    """
    def __init__(self, attack_graph, loss_weight=1.0, k=0):

        g = attack_graph

        # target_logits should be [b, feats], argmax should be [b, feats]
        self.target = attack_graph.graph.target_logits
        self.target_argmax = attack_graph.graph.placeholders.logit_indices

        # Current logits should be [b, feats, chars]
        self.current = tf.transpose(g.victim.raw_logits, [1, 0, 2])

        # Create one hot matrices to multiply by current logits.
        # These essentially act as a filter to keep only the logit we care about
        # for each feature frame

        # reduce the tensor to [b, feats, 1]
        self.logsumexp_c = tf.log(tf.reduce_sum(tf.exp(self.current), axis=2))
        self.logsumexp_t = tf.log(tf.reduce_sum(tf.exp(self.target), axis=2))

        diff = tf.abs(self.target - self.current)

        onehot = tf.one_hot(
            self.target_argmax,
            self.current.shape.as_list()[2],
            on_value=2.0,
            off_value=1.0
        )
        # filter out the other logits
        self.weighted_diff = diff * onehot

        self.diff = tf.reduce_sum(self.weighted_diff, axis=2)

        # kappa is not necessarily > 0 with a beam search decoder => the
        # difference between original + target logits can be negative as the
        # bs decoder cares about the likelihood over *all alignment path steps*.

        # That's is why we repeat the characters when doing searches...

        # also, kappa == 0 will return the same transcription as the original,
        # so the loss would be zero

        # so we take the sign of the kappa value from the logits search and
        # invert the difference to account for each case:

        # k > 0 :=> +1 * (target - current) => target - current
        # k < 0 :=> -1 * (target - current) => current - target
        # k == 0 :=> 0 * (target - current) => 0

        # then we take the maximum between each logit and 0, as per
        # Carlini & Wagner's improved loss function

        signs = tf.sign(attack_graph.graph.placeholders.kappas)

        # penalise characters that are further away than others without causing
        # character differences <0 to become smaller as a result

        self.loss_fn = tf.reduce_sum(self.diff, axis=1) * loss_weight


class AdaptiveKappaCWMaxDiff(object):
    def __init__(self, attack_graph, target_argmax, char_weight=1000.0, k=0.5, loss_weight=1.0, ref_fn=tf.reduce_min):
        """
        This is f_{6} from https://arxiv.org/abs/1608.04644 using the gradient
        clipping update method.

        Difference of:
        - target logits value (B)
        - max other logits value (A -- 2nd most likely)

        Once  B > A, then B is most likely and we can stop optimising.

        Unless -k > B, then k acts as a confidence threshold and continues
        optimisation.

        This will push B to become even less likely.

        N.B. This loss does *not* seems to conform to:
            l(x + d, t) <= 0 <==> C(x + d) = t

        But it does a much better job than the ArgmaxLowConfidence
        implementation as 0 <= l(x + d, t) < 1.0 for a successful decoding

        TODO: This needs testing.
        TODO: normalise to 0 <= x + d <= 1 and convert to tanh space for `change
              of variable` optimisation
        """

        # We have to set k > 0 for this loss function because k = 0 will only
        # cause the probability of the target character to exactly match the
        # next most likely character...
        assert type(k) is float
        assert ref_fn in [tf.reduce_max, tf.reduce_min, tf.reduce_mean]

        g = attack_graph

        # We only use the argmax of the generated alignments so we don't have
        # to worry about finding "exact" alignments
        # target_logits should be [b, feats, chars]
        self.target_argmax = target_argmax  # [b x feats]

        print(self.target_argmax)

        # Current logits is [b, feats, chars]
        # current_argmax is for debugging purposes only
        self.current = tf.transpose(g.victim.raw_logits, [1, 0, 2])
        print(self.current)
        self.current_argmax = tf.argmax(self.current, axis=2)
        print(self.current_argmax)

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

        self.target_logit = tf.reduce_max(self.targ, axis=2)
        self.max_other_logit = tf.reduce_max(self.others, axis=2)

        # Get the min and work out difference between max and min to get a
        # suitable kappa frame-wise
        self.kappa_distrib = self.max_other_logit - ref_fn(self.others, axis=2)

        # each target logit frame must be at least k * other logits difference
        # for loss to minimise.
        self.kappas = self.kappa_distrib * k

        # If target logit is most likely, then the optimiser has done a good job
        # and loss will become negative.
        # Add kappa on the end so that loss is zero when minimised
        self.max_diff_abs = self.max_other_logit - self.target_logit
        self.max_diff = tf.maximum(self.max_diff_abs, -self.kappas) + self.kappas
        self.loss_fn = tf.reduce_sum(self.max_diff, axis=1)

        self.loss_fn = self.loss_fn * loss_weight


class AlignmentLoss(object):
    def __init__(self, alignment_graph):
        seq_lens = alignment_graph.batch.audios.alignment_lengths

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

        per_logit_lengths = batch.audios.alignment_lengths
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
        self.loss = self.graph.adversarial_loss

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

        logits = v.get_logits(v.raw_logits, b.feeds.examples)
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
                feed_dict=b.feeds.alignments
            )

            decodings, probs = victim.inference(
                b,
                logits=softmax,
                decoder="batch",
                top_five=False
            )

            if all([d == b.targets.phrases[0] for d in decodings]):
                s = "Found an alignment for each example:"
                for d, p, t in zip(decodings, probs, b.targets.phrases):
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
