import tensorflow as tf
import numpy as np
import os

from copy import deepcopy

from cleverspeech.data import ETL
from cleverspeech.data import Feeds
from cleverspeech.data import Generators
from cleverspeech.data.Results import SingleJsonDB
from SecEval import VictimAPI as DeepSpeech
from cleverspeech.utils.Utils import log, l_map
from cleverspeech.utils.RuntimeUtils import create_tf_runtime, AttackSpawner

GPU_DEVICE = 0
MAX_PROCESSES = 1
SPAWN_DELAY = 15

INDIR = "./samples/all/"
TARGETS_PATH = "./samples/cv-valid-test.csv"
OUTDIR = "./target-logits/simple-branched-grid-search/"

TOKENS = " abcdefghijklmnopqrstuvwxyz'-"

BATCH_SIZE = 10
NUMB_EXAMPLES = 1000
MAX_AUDIO_LENGTH = 140000
TARGETS_POOL = 1000
AUDIO_EXAMPLES_POOL = 2000

MAX_KAPPA = 20
MAX_DEPTH = 3


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


def gen_new_indices(new_target, n_feats, repeats):

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


def add_kappa_to_all_logits(n_feats, n_padded_feats, example_logits, new_target, kappa, repeats):
    """
    Make the required modifications to the logits obtained from the original
    example.

    This search algorithm does:

    1. generate a vector with the desired argmax for our target phrase
    2. for each time step, increase the value of the desired character by kappa
    3. do any padding. N.B. this is never called for now.

    :param n_feats: number of features/time steps.
    :param n_padded_feats: number of features for the longest example in batch
    :param example_logits: the original logits for the current example
    :param new_target: the modified target phrase -- i.e. with extra blanks
    :param kappa: how much increase to apply to the logit values for characters
    :param repeats: how many repeated characters we want to insert

    :return: new logits for the current example which will now decode to the
    target phrase
    """
    padding = n_padded_feats - example_logits.shape[0]

    new_argmax = np.asarray(
        l_map(
            lambda x: x, gen_new_indices(new_target, n_feats, repeats)
        ),
        dtype=np.int32
    )

    for l, a in zip(example_logits, new_argmax):
        if np.argmax(l) == a:
            # where we our target character is already the most likely we want
            # to make sure it's greater than the next value by kappa
            # l[a] = l[a] + kappa
            # sorted_indices = np.argsort(l)
            # sorted_values = l[sorted_indices]
            # second_largest = sorted_values[-2]
            # if l[a] > second_largest:
            #     pass
            # else:
            #     l[a] = sorted_values[-2] + kappa
            pass
        #else:
        #    # otherwise just increase the class we want by kappa
        l[a] = l[a] + kappa

    # we never actually call padding as we work out the appropriate length
    # based on the real number of features (not the batch's number of features).
    if padding > 0:
        padd_arr = np.zeros([29])
        padd_arr[28] = kappa
        padding = [padd_arr for _ in range(padding)]
        example_logits = np.concatenate(example_logits, np.asarray(padding))

    return example_logits, new_argmax


def get_original_decoder_db_outputs(model, batch):
    """
    Get the decoding and transcription confidence score for the original example

    :param model: the model we attack, must have an inference() method
    :param batch: the batch of examples to try to generate logits for

    :return:
    """
    decodings, probs = model.inference(
        batch,
        feed=batch.feeds.examples,
        decoder="batch",
        top_five=False
    )

    return decodings, probs


def get_original_network_db_outputs(model, batch):
    """
    Return the logits of the encoder (DeepSpeech's RNN).

    :param model: the model we attack, must have an inference() method
    :param batch: the batch of examples to try to generate logits for

    :return original_logits: the first set of logits we found.
    """

    logits, softmax = model.get_logits(
        [
            tf.transpose(model.raw_logits, [1, 0, 2]),
            model.logits
        ],
        batch.feeds.examples
    )
    model.reset_state()

    return logits, softmax


def check_logits_variation(model, batch, original_logits):
    """
    Model is fixed and Logits should *never* change between inference calls for
    the same audio example. If it does then something is wrong.

    :param model: the model we attack, must have an inference() method
    :param batch: the batch of examples to try to generate logits for
    :param original_logits: the first set of logits we found.

    :return: Nothing.
    """
    logits, smax = get_original_network_db_outputs(
        model, batch
    )

    assert np.sum(logits - original_logits) == 0.0


def branched_grid_search(sess, model, batch, original_probs, real_logits):

    for idx in range(batch.size):

        basename = batch.audios.basenames[idx]
        target_phrase = batch.targets.phrases[idx]
        indices = batch.targets.indices[idx]
        n_padded_feats = batch.audios.feature_lengths[idx]
        n_feats = batch.audios.alignment_lengths[idx] + 1
        targs_id = batch.targets.ids[idx]
        original_audio = batch.audios.audio[idx]
        absolute_file_path = os.path.abspath(
            os.path.join(INDIR, basename)
        )

        new_target_phrases = np.array(
            insert_target_blanks(indices),
            dtype=np.int32
        )
        current_repeat = 2

        #while current_repeat * len(new_target_phrases) <= n_feats:

        if n_feats < len(new_target_phrases):
            s = "Skipping:   {b} for {r} repeats and target {t}".format(
                b=basename,
                r=current_repeat,
                t=targs_id
            )
            s += " (not enough space)."
            log(s, wrap=False)

        else:
            current_repeat = n_feats // len(new_target_phrases)

            new_indices = np.asarray(
                l_map(
                    lambda x: x,
                    gen_new_indices(
                        new_target_phrases,
                        n_feats,
                        current_repeat
                    )
                ),
                dtype=np.int16
            )

            # s = "Processing: {b} for {r} repeats and target {t}".format(
            #     b=basename,
            #     r=current_repeat,
            #     t=targs_id
            # )
            # log(s, wrap=False)

            kappa = 0
            current_depth = 1
            max_depth = 10 ** MAX_DEPTH

            result = {
                "kappa": kappa,
                "decoding": None,
                "score": 0.0,
                "new_logits": None,
                "new_softmax": None,
                "argmax": None,
                "audio_filepath": absolute_file_path,
                "audio_basename": basename,
                "audio_data": original_audio,
                "repeats": current_repeat,
                "original_score": original_probs[idx],
                "n_feats": n_feats,
                "targ_id": targs_id,
                "target_phrase": target_phrase,
                "new_target": new_target_phrases,
                "original_logits": real_logits[idx][:n_feats]
            }

            alignment = deepcopy(real_logits[idx])

            for char_idx, character in enumerate(new_target_phrases):

                for repeat_idx in range(0, current_repeat):
                    alignment_idx = char_idx * current_repeat + repeat_idx
                    print("IDX: ", alignment_idx, "NFEATS: ", n_feats)

                    subseq_diff = np.max(alignment[alignment_idx]) - np.min(alignment[alignment_idx])
                    subseq_diff = np.abs(subseq_diff)
                    subseq_diff = (np.round(subseq_diff) // 5) + 5

                    kappa = 0

                    while kappa <= subseq_diff:

                        # make some logits!

                        alignment[alignment_idx:, character] += 5

                        #print(alignment, subsequence)

                        new_logits = np.asarray([alignment])

                        # TODO: Test that only one character per frame has changed
                        #  and the remainder are the same as before (i.e. diff = 0)

                        # TODO: how can we escape the need to run as
                        new_softmaxes = sess.run(
                            tf.nn.softmax(new_logits)
                        )

                        decodings, probs = model.inference(
                            batch,
                            logits=np.asarray(new_softmaxes),
                            decoder="ds",
                            top_five=False
                        )

                        # decodings = model.tf_beam_decode(
                        #     sess,
                        #     tf.transpose(new_logits, [1, 0, 2]),
                        #     [n_feats],
                        #     TOKENS
                        # )[0]
                        if char_idx == 28:
                            current_character_decode = decodings[:char_idx - 1]
                            current_character_target = target_phrase[:char_idx - 1]
                        else:
                            current_character_decode = decodings[:char_idx]
                            current_character_target = target_phrase[:char_idx]

                        # print(
                        #     character,
                        #     kappa,
                        #     new_logits.shape,
                        #     alignment.shape,
                        #     element.shape,
                        #     current_character_decode,
                        #     "|",
                        #     current_character_target,
                        #     #subsequence
                        # )

                        if current_character_decode == current_character_target:
                            # print("OOOOOOOOOOOO YEAH")
                            kappa = subseq_diff + 1

                        else:
                            kappa += 5

                print("current probs : {p:.1f} decode: {d}".format(p=probs, d=decodings[:char_idx]))
                print("previous score: {p:.1f} target: {d} ".format(p=original_probs[idx], d=target_phrase[:char_idx]))

            print("decode: ", decodings)
            print("target: ", target_phrase)
            print("probs", probs)



def write_results(result):
    # TODO: skip later repeats if *NOTHING* was successful

    # store the db_outputs in a json file with the
    # absolute filepath *and* the original example as it
    # makes loading data for the actual optimisation attack
    # a hell of a lot easier
    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)

    db_path = "{b}_targid-{t}".format(
        b=result["audio_basename"].rstrip(".wav"),
        t=result["targ_id"]
    )

    example_db = SingleJsonDB(OUTDIR)
    example_db.open(db_path).put(result)

    # print out how we've done
    s = "Success:    {b} for {r} repeats and target {t}".format(
        b=result["audio_basename"],
        r=result["repeats"],
        t=result["targ_id"],
    )
    s += " kappa: {k} orig score: {o:.1f} new score: {n:.1f}".format(
        k=result["kappa"],
        o=result["original_score"],
        n=result["score"],
    )
    s += " logits diff: {:.0f}".format(
        np.sum(result["original_logits"] - result["new_logits"])
    )
    s += " Wrote to: {}".format(db_path)
    log(s, wrap=False)


def run(batch):

    tf_session, tf_device = create_tf_runtime()

    with tf_session as sess, tf_device:

        ph_examples = tf.placeholder(
            tf.float32, shape=[batch.size, batch.audios.max_length]
        )
        ph_lens = tf.placeholder(
            tf.float32, shape=[batch.size]
        )

        model = DeepSpeech.Model(
            sess, ph_examples, batch, tokens=TOKENS, beam_width=500
        )

        batch.feeds.create_feeds(
            ph_examples, ph_lens
        )

        original_decodings, original_probs = get_original_decoder_db_outputs(
            model, batch
        )

        real_logits, real_softmax = get_original_network_db_outputs(
            model, batch
        )
        for _ in range(10):
            check_logits_variation(
                model, batch, real_logits
            )

        search_gen = branched_grid_search(
            sess,
            model,
            batch,
            original_probs,
            real_logits,
        )

        for idx, result in search_gen:

            write_results(result)


def main():

    assert MAX_DEPTH > 1.0 and MAX_DEPTH % 1 == 0

    audio_fps_etl = ETL.AllAudioFilePaths(
        INDIR,
        AUDIO_EXAMPLES_POOL,
        sort_by_file_size="desc",
        filter_term=".wav",
        max_samples=MAX_AUDIO_LENGTH
    )
    all_audio_file_paths = audio_fps_etl.extract().transform().load()

    transcriptions_etl = ETL.AllTargetPhrases(
        TARGETS_PATH, TARGETS_POOL
    )
    all_transcriptions = transcriptions_etl.extract().transform().load()

    # Generate the batches in turn, rather than all in one go ...

    batch_factory = Generators.BatchGenerator(
        all_audio_file_paths, all_transcriptions, BATCH_SIZE,
    )

    # ... To save resources by only running the final ETLs on a batch of data

    batch_gen = batch_factory.generate(
        ETL.AudioExamples, ETL.TargetPhrases, Feeds.Validation
    )

    # Create the factory we'll use to iterate over N examples at a time.
    attack_spawner = AttackSpawner(
        gpu_device=GPU_DEVICE,
        max_processes=MAX_PROCESSES,
        delay=SPAWN_DELAY
    )
    with attack_spawner as spawner:
        for (b_id, batch) in batch_gen:
            spawner.spawn(run, batch)


if __name__ == '__main__':
    main()
