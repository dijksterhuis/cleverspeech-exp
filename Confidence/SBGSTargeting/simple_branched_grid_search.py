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
MAX_PROCESSES = 5
SPAWN_DELAY = 15

SEARCH_METHOD = "sbgs"

INDIR = "./samples/all/"
TARGETS_PATH = "./samples/cv-valid-test.csv"
OUTDIR = os.path.join("./adv/", SEARCH_METHOD + "/")

TOKENS = " abcdefghijklmnopqrstuvwxyz'-"

BATCH_SIZE = 10
NUMB_EXAMPLES = 1000
MAX_AUDIO_LENGTH = 140000
TARGETS_POOL = 500
AUDIO_EXAMPLES_POOL = 2000

MIN_KAPPA = -5.0
MAX_DEPTH = 3


def entropy(batch_softmaxes):
    return np.max(- np.sum(batch_softmaxes * np.log(batch_softmaxes), axis=1), axis=1)


def update_kappa(kappa, current_depth, max_depth):
    kappa = kappa + 1 / current_depth
    kappa = np.round(kappa, int(np.log10(max_depth)))
    return kappa


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


def branched_grid_search(sess, model, batch, original_probs, real_logits, reference=np.max):

    for idx in range(batch.size):

        basename = batch.audios.basenames[idx]
        target_phrase = batch.targets.phrases[idx]
        indices = batch.targets.indices[idx]
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
        max_repeats = n_feats // len(new_target_phrases)
        for current_repeat in range(0, max_repeats):

            # s = "Processing: {b} for {r} repeats and target {t}".format(
            #     b=basename,
            #     r=current_repeat,
            #     t=targs_id
            # )
            # log(s, wrap=False)

            kappa = MIN_KAPPA
            current_depth = 1
            max_depth = 10 ** MAX_DEPTH

            result = {
                "kappa": kappa,
                "decoding": None,
                "score": float('inf'),
                "spc": float('inf'),
                "new_logits": None,
                "new_softmax": None,
                "argmax": None,
                "audio_filepath": absolute_file_path,
                "audio_basename": basename,
                "audio_data": original_audio,
                "repeats": current_repeat,
                "original_score": original_probs[idx],
                "original_spc": original_probs[idx] / len(target_phrase),
                "n_feats": n_feats,
                "targ_id": targs_id,
                "target_phrase": target_phrase,
                "new_target": new_target_phrases,
                "original_logits": real_logits[idx][:n_feats]
            }

            while current_depth <= max_depth:

                # make some logits!

                # I don't trust np.copy() so I call deepcopy().
                # there's a danger of memory leaks, but the attack process
                # spawner deals with that for us.
                new_logits = deepcopy(real_logits[idx])

                alignment_indices = np.asarray(
                    l_map(
                        lambda x: x,
                        gen_new_indices(
                            new_target_phrases,
                            n_feats,
                            current_repeat
                        )
                    ),
                    dtype=np.int32
                )

                pad_dims = new_logits.shape[0] - alignment_indices.shape[0]
                padding = np.ones([pad_dims], dtype=np.int32)
                padding *= len(TOKENS) - 1

                alignment_indices = np.concatenate(
                    [alignment_indices, padding]
                )

                for l, a in zip(new_logits, alignment_indices):
                    l[a] = reference(l) + kappa

                new_logits = np.asarray([new_logits])

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
                score_per_char = probs / len(target_phrase)

                # scores increase as the token probabilities get
                # closer. this seems counter intuitive, but it works

                current_decoding_correct = decodings == target_phrase
                current_score_better = result["spc"] > score_per_char
                best_score_non_zero = result["score"] != float('inf')

                if current_decoding_correct and current_score_better:

                    # great success!
                    best_kappa = kappa
                    kappa = update_kappa(kappa, current_depth, max_depth)

                    result["kappa"] = best_kappa
                    result["decoding"] = decodings
                    result["score"] = probs
                    result["spc"] = score_per_char
                    result["new_softmax"] = new_softmaxes[:n_feats]
                    result["new_logits"] = new_logits[0][:n_feats]
                    result["argmax"] = alignment_indices[:n_feats]

                elif best_score_non_zero:
                    # we have been successful at some point, so
                    # reduce the search depth

                    d = current_depth * 10

                    # there's a weird bug where search depths
                    # become 0, and then kappa start to tend
                    # toward negative infinity, e.g.
                    # kappa: -0.11 -> -0.111 -> -0.11 -> -inf
                    if d == 0:
                        # something went wrong...
                        current_depth = max_depth

                    elif d > max_depth:
                        # we've hit the maximum, update depths,
                        # but don't update kappa
                        break

                    else:
                        # we're not at maximum search depth, so we
                        # must have just seen something good so
                        # change the depth and update kappa
                        current_depth = d
                        kappa = result["kappa"]
                        kappa = update_kappa(kappa, current_depth, max_depth)

                # elif kappa >= MIN_KAPPA:
                #     # we've hit a boundary condition
                #     break
                else:
                    # we haven't found anything yet
                    kappa = update_kappa(kappa, current_depth, max_depth)

            best_decoding_check = result["decoding"] != target_phrase
            best_spc_check = result["spc"] >= result["original_spc"]

            if best_decoding_check:
                # we've not been successful, increase the number of repeats
                # and try again
                s = "Failure:    {b} for {r} repeats and target {t}".format(
                    b=result["audio_basename"],
                    r=result["repeats"],
                    t=result["targ_id"],
                )
                s += " (decoding does not match target phrase)."
                log(s, wrap=False)

            elif best_spc_check:
                # we've not been successful, increase the number of repeats
                # and try again
                s = "Failure:    {b} for {r} repeats and target {t}".format(
                    b=result["audio_basename"],
                    r=result["repeats"],
                    t=result["targ_id"],
                )
                s += " adversarial score per char. <= original score per char.:"
                s += " {a:.1f} vs. {b:.1f}".format(
                    a=result["spc"],
                    b=result["original_spc"]
                )
                log(s, wrap=False)

            else:
                yield idx, result


def write_results(result):
    # TODO: skip later repeats if *NOTHING* was successful

    # store the db_outputs in a json file with the
    # absolute filepath *and* the original example as it
    # makes loading data for the actual optimisation attack
    # a hell of a lot easier
    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)

    # write out for each repeat value *and* the last success (is most confident)
    db_path = "{b}_targid-{t}_rep-{r}".format(
        b=result["audio_basename"].rstrip(".wav"),
        t=result["targ_id"],
        r=result["repeats"],
    )
    example_db = SingleJsonDB(OUTDIR)
    example_db.open(db_path).put(result)

    db_path = "{b}_targid-{t}_latest".format(
        b=result["audio_basename"].rstrip(".wav"),
        t=result["targ_id"],
    )
    example_db = SingleJsonDB(OUTDIR)
    example_db.open(db_path).put(result)

    # log how we've done
    s = "Success:    {b} for {r} repeats and target {t}".format(
        b=result["audio_basename"],
        r=result["repeats"],
        t=result["targ_id"],
    )
    s += " kappa: {k:.3f} orig score p.c.: {o:.1f} new score p.c.: {n:.1f}".format(
        k=result["kappa"],
        o=result["original_spc"],
        n=result["spc"],
    )
    s += " orig score : {o:.1f} new score: {n:.1f}".format(
        o=result["original_score"],
        n=result["score"],
    )
    s += " logits diff: {:.0f}".format(
        np.abs(np.sum(result["original_logits"] - result["new_logits"]))
    )
    s += " entropy: {}".format(
        entropy(result["new_softmax"])
    )

    s += " Wrote targeting data."
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

        print(real_softmax.shape)

        s = "Current Entropy => " + "".join(["\t{:.6f}".format(x) for x in entropy(real_softmax)])
        log(s)

        # reset the batch's targets to the original decodings
        # ==> we want to see how much more confident they can be made.
        # TODO: New Generator to pair targets with audios.
        batch.targets.phrases = original_decodings

        batch.targets.indices = [
            np.asarray([TOKENS.index(x) for x in decoding])
            for decoding in original_decodings
        ]

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
