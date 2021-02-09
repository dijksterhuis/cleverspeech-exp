import os
import json
import numpy as np

from cleverspeech.data.Batches import Batch
from cleverspeech.data.ETL import ETL, TargetPhrases
from cleverspeech.data.Generators import BaseBatchGenerator, pop_target_phrase
from cleverspeech.utils.Utils import np_arr, np_zero, lcomp, load_wavs, l_map, assert_positive_int


class TargetLogitsBatch(object):
    def __init__(self, data, tokens, maxlen):
        self.tokens = tokens
        self.maximum_length = maxlen
        [
            self.logits,
            self.filtered_logits,
            self.alignment_indices,
            self.softmax,
            self.phrases,
            self.ids,
            self.lengths,
            self.repeats,
            self.kappa,
        ] = data


class AllTargetLogitsAndAudioFilePaths(ETL):
    def __init__(self, indir, n, filter_term=None, sort_by_file_size=True):

        # private vars
        self.__data = indir
        self.__filter_term = filter_term
        self.__sort_file_size = sort_by_file_size
        self.__extracted = None
        self.__transformed = None

        # public vars
        self.numb = n

    def extract(self):

        def open_json(fp):
            with open(fp, 'r') as f:
                x = json.load(f)[0]

            out = [
                x["audio_filepath"][0],
                x["audio_basename"][0],
                x["new_logits"],
                x["new_softmax"],
                x["argmax"],
                x["decoding"][0],
                x["targ_id"][0],
                x["n_feats"][0],
                x["repeats"][0],
                x["kappa"][0],
            ]
            return out

        file_paths = l_map(lambda x: x, self.get_file_paths(self.__data))

        if self.__filter_term is not None:
            file_paths = list(
                filter(
                    lambda x: self.__filter_term in x[2], file_paths
                )
            )

        if self.__sort_file_size is not None:
            file_paths.sort(key=lambda x: x[0], reverse=False)

        if self.numb <= len(file_paths):
            file_paths = file_paths[:self.numb]

        parsed_json = l_map(lambda x: open_json(x[1]), file_paths)

        self.__extracted = parsed_json

        return self

    def transform(self):
        self.__transformed = self.__extracted
        return self

    def load(self):
        return self.__transformed


class TargetLogits(ETL):
    def __init__(self, data, tokens=" abcdefghijklmnopqrstuvwxyz'-"):
        # private_vars
        self.__data = data
        self.__extracted = None
        self.__transformed = None

        # public vars
        self.tokens = tokens
        self.maximum_length = None

        super().__init__()

    def extract(self):
        """
                x["new_logits"][0],
                x["new_softmax"][0],
                x["argmax"][0],
                x["decoding"][0],
                x["targ_id"][0],
                x["n_feats"][0],
                x["repeats"][0],
                x["kappa"][0],


        :return: None
        """
        target_logits = l_map(lambda x: np_arr(x[0], np.float32), self.__data)
        target_softmax = l_map(lambda x: np_arr(x[1], np.float32), self.__data)
        target_indices = l_map(lambda x: np_arr(x[2], np.int32), self.__data)

        target_phrases = l_map(lambda x: x[3], self.__data)
        ids = l_map(lambda x: x[4], self.__data)
        n_feats = l_map(lambda x: x[5], self.__data)
        repeats = l_map(lambda x: x[6], self.__data)
        kappas = l_map(lambda x: x[7], self.__data)

        self.__extracted = [
            target_logits,
            target_indices,
            target_softmax,
            target_phrases,
            ids,
            n_feats,
            repeats,
            kappas,
        ]

        return self

    def transform(self):

        def padding(x, n_feat, max_len):
            x = x[:int(n_feat)]
            pad = np.zeros([max_len - x.shape[0], len(self.tokens)])
            return np.concatenate((x, pad), axis=0)

        def make_kappa_non_zero(x):
            if x == 0:
                return 1e-8
            else:
                return x

        def pad_indices(x, n_feat, max_len):
            x = x[:int(n_feat)]
            pad = np.zeros([max_len - x.shape[0]])
            return np.concatenate((x, pad), axis=0)

        [
            target_logits,
            target_indices,
            target_softmax,
            target_phrases,
            ids,
            n_feats,
            repeats,
            kappas,
        ] = self.__extracted

        n_feats = l_map(lambda x: x - 1, n_feats)

        self.maximum_length = int(max(n_feats))

        target_indices = np_arr(
            l_map(
                lambda x: pad_indices(x[0], x[1], self.maximum_length),
                zip(target_indices, n_feats)
            ),
            np.int32
        )

        for align in target_indices:
            assert len(align) == self.maximum_length

        target_logits = np_arr(
            l_map(
                lambda x: padding(x[0], x[1], self.maximum_length),
                zip(target_logits, n_feats)
            ),
            np.float32
        )

        filtered_logits = np.take(target_logits, target_indices)

        kappas = l_map(
            lambda x: [make_kappa_non_zero(x)],
            kappas
        )

        self.__transformed = [
            target_logits,
            filtered_logits,
            target_indices,
            target_softmax,
            target_phrases,
            ids,
            n_feats,
            repeats,
            kappas,
        ]

        return self

    def load(self):
        """
        TargetLogits =
            self.logits,
            self.filtered_logits,
            self.alignment_indices,
            self.softmax,
            self.phrases,
            self.ids,
            self.lengths,
            self.repeats,
            self.kappa,
        :return:
        """

        return TargetLogitsBatch(self.__transformed, self.tokens, self.maximum_length)


class LogitsBatchGenerator:
    def __init__(self, all_data, batch_size):

        numb_examples = len(all_data)

        assert batch_size > 0 and type(batch_size) is int
        assert numb_examples > 0 and type(numb_examples) is int

        self.size = batch_size
        self.numb_examples = numb_examples

        self.all_data = all_data

        self.batch = None
        self.id = None

    def generate(self, audio_etl_cls, target_etl_cls, feeds_cls):

        for idx in range(0, self.numb_examples, self.size):

            # Handle remainders: number of examples // desired batch size != 0
            if len(self.all_data) < self.size:
                self.size = len(self.all_data)

            # get the next batch of data
            batch_data = l_map(
                lambda x: self.all_data.pop(x-1),
                range(self.size, 0, -1)
            )

            # split into audio vs. targeting data at the last possible
            # opportunity to ensure we have the always get the relevant entries.
            audios = l_map(
                lambda x: (None, x[0], x[1]),
                batch_data
            )
            targets = l_map(
                lambda x: x[2:],
                batch_data
            )

            # actually load the n data examples as a batch
            audios = audio_etl_cls(audios)
            targets = target_etl_cls(targets)

            self.batch = Batch(
                self.size,
                audios.extract().transform().load(),
                targets.extract().transform().load(),
                feeds_cls
            )

            self.id = idx

            yield self.id, self.batch


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
                    create_new_target_indices(x, y, z)
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
