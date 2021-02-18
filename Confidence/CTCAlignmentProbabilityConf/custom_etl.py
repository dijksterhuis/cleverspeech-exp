import numpy as np

from cleverspeech.data.Batches import Batch
from cleverspeech.data.ETL import TargetPhrases
from cleverspeech.data.Generators import BaseBatchGenerator, pop_target_phrase
from cleverspeech.utils.Utils import np_arr, l_map
from cleverspeech.utils.Utils import assert_positive_int


def calculate_possible_repeats(actual_n_feats, target_phrase_length):
    return actual_n_feats // target_phrase_length


def insert_target_blanks(target_indices):
    # get a shifted list so we can compare back one step in the phrase

    previous_indices = target_indices.tolist()
    previous_indices.insert(0, None)

    # insert blank tokens where ctc would expect them - i.e. `do-or`
    # also insert a blank at the start (it gives space for the RNN to "warm up")
    with_repeats = [28]
    for current, previous in zip(target_indices, previous_indices):
        if not previous:
            with_repeats.append(current)
        elif current == previous:
            with_repeats.append(28)
            with_repeats.append(current)
        else:
            with_repeats.append(current)
    return with_repeats


def create_new_target_indices(new_target, n_feats, repeats):

    """
    Taking into account the space we have available, find out the new argmax
    indices for each frame of audio which relate to our target phrase

    :param new_target: the new target phrase included additional blank tokens
    :param n_feats: the number of features in the logits (time steps)
    :param repeats: the number of repeats for each token

    :return: the index for each frame in turn
    """

    spacing = n_feats // new_target.size

    for t in new_target:
        for i in range(spacing):
            if i > repeats:
                yield 28
            else:
                yield t


def pad_indices(indices, act_len):
    n_paddings = act_len - len(indices)
    padded = np.concatenate([indices, np.ones(n_paddings + 1) * 28])
    return padded


class CTCHiScoresBatchGenerator(BaseBatchGenerator):
    def __init__(self, all_audio_file_paths, all_targets, batch_size):

        numb_examples = len(all_audio_file_paths)
        numb_targets = len(all_targets)

        assert_positive_int(batch_size)
        assert_positive_int(numb_examples)
        assert_positive_int(numb_targets)

        assert numb_targets * batch_size >= numb_examples

        self.all_audio_file_paths = all_audio_file_paths
        self.all_targets = all_targets

        super().__init__(batch_size, numb_examples)

    def generate(self, audio_etl_cls, target_etl_cls, feeds_cls):

        for idx in range(0, self.numb_examples, self.size):

            # Handle remainders: number of examples // desired batch size != 0
            if len(self.all_audio_file_paths) < self.size:
                self.size = len(self.all_audio_file_paths)

            # get n files paths and create the audio batch data
            audio_batch_data = self.popper(self.all_audio_file_paths)
            audios = audio_etl_cls(audio_batch_data)
            audio_batch = audios.extract().transform().load()

            #  we need to make sure target phrase length < n audio feats.
            # also, the first batch should also have the longest target phrase
            # and longest audio examples so we can easily manage GPU Memory
            # resources with the AttackSpawner context manager.

            actual_n_feats = audio_batch.alignment_lengths

            # ensure we get *at least* duplicate characters by dividing the
            # minimum possible length by 3
            target_phrase = pop_target_phrase(
                self.all_targets, min(actual_n_feats) // 3
            )

            # each target must be the same length else numpy throws a hissyfit
            # because it can't understand skewed matrices
            target_batch_data = l_map(
                lambda _: target_phrase, range(self.size)
            )

            # actually load the n phrases as a batch of target data
            targets = target_etl_cls(target_batch_data, actual_n_feats)
            target_batch = targets.extract().transform().load()

            self.batch = Batch(
                self.size,
                audio_batch,
                target_batch,
                feeds_cls
            )

            self.id = idx

            yield self.id, self.batch


class RepeatsTargetPhrases(TargetPhrases):
    def __init__(self, data, n_feats, tokens=" abcdefghijklmnopqrstuvwxyz'-"):

        self.__n_feats = n_feats
        print("N", n_feats)
        super().__init__(data, tokens=tokens)

    def extract(self):
        return super().extract()

    def transform(self):
        super().transform()

        # hacky access to parent class
        transformed = getattr(self, "_TargetPhrases__transformed")

        new_indices = np_arr(
            l_map(
                lambda x: insert_target_blanks(x),
                transformed[2]
            ),
            np.int32
        )
        # calculate the actual number of repeats
        z = zip(self.__n_feats, transformed[3])
        n_repeats = [calculate_possible_repeats(x, y) for x, y in z]

        # do linear expansion only on the existing indices (target phrases
        # are still valid as they are).
        z = zip(new_indices, self.__n_feats, n_repeats)
        transformed[2] = np_arr(
            [
                l_map(
                    lambda x: x,
                    create_new_target_indices(x, y, z, )
                ) for x, y, z in z
            ],
            np.int32
        )

        # do padding for nopn-ctc loss
        z = zip(transformed[2], self.__n_feats)
        transformed[2] = np_arr(
            [
                l_map(
                    lambda x: x,
                    pad_indices(x, y)
                ) for x, y in z
            ],
            np.int32
        )

        # update the target sequence lengths
        transformed[3] = l_map(
            lambda x: x.size,
            transformed[2]
        )

        # hacky access to parent class
        setattr(self, "_TargetPhrases__transformed", transformed)

        return self

    def load(self):
        return super().load()


