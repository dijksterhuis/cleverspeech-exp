import tensorflow as tf
import numpy as np
import sys
import os

from cleverspeech.Data import Batches
from cleverspeech.Data import Generators
from cleverspeech.Data.Results import SingleJsonDB
from DeepSpeechSecEval import VictimAPI as DeepSpeech
from cleverspeech.Utils import log, l_map
from cleverspeech.RuntimeUtils import create_tf_runtime, AttackSpawner

INDIR = "./samples/"
OUTDIR = "./experiments/StrongAlignments/target-logits"

TOKENS = " abcdefghijklmnopqrstuvwxyz'-"

BATCH_SIZE = 1
NUMB_EXAMPLES = 1000
TARGETS_POOL = 2000
AUDIO_EXAMPLES_POOL = 2000

MAX_REPEATS = 10
MAX_KAPPA = 9.0
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

    new_argmax = l_map(
        lambda x: x, gen_new_indices(new_target, n_feats, repeats)
    )

    for l, a in zip(example_logits, new_argmax):
        if np.argmax(l) == a:
            # where we our target character is already the most likely we don't
            # have to do any work
            # l[a] = l[a] + kappa
            pass
        else:
            # otherwise just increase the class we want by kappa
            l[a] = np.max(l) + kappa

    # we never actually call padding as we work out the appropriate length
    # based on the real number of features (not the batch's number of features).
    if padding > 0:
        padd_arr = np.zeros([29])
        padd_arr[28] = kappa
        padding = [padd_arr for _ in range(padding)]
        example_logits = np.concatenate(example_logits, np.asarray(padding))

    return example_logits


def add_repeat_kappa_to_all_logits(n_feats, n_padded_feats, example_logits, new_target, kappas_dict, repeats):
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

    new_argmax = l_map(
        lambda x: x, gen_new_indices(new_target, n_feats, repeats)
    )

    for l, a in zip(example_logits, new_argmax):
        if np.argmax(l) == a:
            # where we our target character is already the most likely we don't
            # have to do any work
            # l[a] = l[a] + kappa
            pass
        else:
            # otherwise just increase the class we want by kappa
            l[a] = np.max(l) + kappas_dict[a]

    # we never actually call padding as we work out the appropriate length
    # based on the real number of features (not the batch's number of features).
    if padding > 0:
        padd_arr = np.zeros([29])
        padd_arr[28] = kappas_dict[28]
        padding = [padd_arr for _ in range(padding)]
        example_logits = np.concatenate(example_logits, np.asarray(padding))

    return example_logits


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
            insert_target_blanks(indices)
        )

        for current_repeat in range(MAX_REPEATS):

            log("", wrap=True)

            s = "Processing: {b} for {r} repeats and target {t}".format(
                b=basename,
                r=current_repeat,
                t=targs_id
            )
            log(s, wrap=True)

            kappa = MAX_KAPPA
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

            while current_depth <= max_depth:

                if len(new_target_phrases) * current_repeat > n_feats:
                    # repeats won't fit logits space so completely
                    # skip any further processing.
                    break

                else:
                    # otherwise, make some logits!
                    initial_logits = real_logits[idx].copy()

                    new_logits = add_kappa_to_all_logits(
                        n_feats,
                        n_padded_feats,
                        initial_logits,
                        new_target_phrases,
                        kappa,
                        current_repeat
                    )

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

                s = "\r"
                s += "Current kappa: {}".format(kappa)
                s += "\tCurrent probs: {}".format(probs)
                s += "\t"
                # s += "\tCurrent Depth: "+"\t".join(
                #         l_map(lambda x: str(x), new_search_depth)
                #     )
                sys.stdout.write(s)
                sys.stdout.flush()

                # scores increase as the token probabilities get
                # closer. this seems counter intuitive, but it works

                if decodings == target_phrase and result["score"] < probs:

                    # great success!

                    best_kappa = kappa
                    kappa = kappa - 1 / current_depth
                    kappa = np.round(
                        kappa,
                        int(np.log10(max_depth))
                    )

                    # store the db_outputs
                    argmax = "".join(
                        l_map(
                            lambda x: TOKENS[x],
                            np.argmax(
                                new_logits[0][:n_feats],
                                axis=1
                            )
                        )
                    )

                    result["kappa"] = best_kappa
                    result["decoding"] = decodings
                    result["score"] = probs
                    result["new_softmax"] = new_softmaxes[:n_feats]
                    result["new_logits"] = new_logits[0][:n_feats]
                    result["argmax"] = argmax

                elif result["score"] != 0.0:
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
                        kappa = kappa - 1 / current_depth
                        kappa = np.round(
                            kappa,
                            int(np.log10(max_depth))
                        )

                elif kappa <= -MAX_KAPPA:
                    # we've hit a minimum boundary condition
                    break
                else:
                    # we haven't found anything yet
                    kappa = kappa - 1 / current_depth
                    kappa = np.round(
                        kappa,
                        int(np.log10(max_depth))
                    )

            if result["decoding"] != target_phrase:
                # we've not been successful, we probably won't
                # find anything useful by increasing the number of
                # repeats so break out of the loop
                break

            else:
                yield idx, result


def bisection_search(sess, model, batch, original_probs, real_logits):

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
            insert_target_blanks(indices)
        )

        for current_repeat in range(MAX_REPEATS):

            log("", wrap=True)

            s = "Processing: {b} for {r} repeats and target {t}".format(
                b=basename,
                r=current_repeat,
                t=targs_id
            )
            log(s, wrap=True)

            main_kappa = MAX_KAPPA

            big_kappa, small_kappa = main_kappa, -main_kappa
            current_depth = 1
            max_depth = 10 ** MAX_DEPTH

            result = {
                "kappa": main_kappa,
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

            while current_depth <= max_depth:

                if len(new_target_phrases) * current_repeat > n_feats:
                    # repeats won't fit logits space so completely
                    # skip any further processing.
                    break

                else:
                    # otherwise, make some logits!
                    initial_logits = real_logits[idx].copy()

                    big_new_logits = add_kappa_to_all_logits(
                        n_feats,
                        n_padded_feats,
                        initial_logits,
                        new_target_phrases,
                        big_kappa,
                        current_repeat
                    )
                    small_new_logits = add_kappa_to_all_logits(
                        n_feats,
                        n_padded_feats,
                        initial_logits,
                        new_target_phrases,
                        small_kappa,
                        current_repeat
                    )

                big_new_logits = np.asarray([big_new_logits])
                small_new_logits = np.asarray([small_new_logits])

                # TODO: Test that only one character per frame has changed
                #  and the remainder are the same as before (i.e. diff = 0)

                # TODO: how can we escape the need to run as
                big_new_softmaxes = sess.run(
                    tf.nn.softmax(big_new_logits)
                )
                small_new_softmaxes = sess.run(
                    tf.nn.softmax(small_new_logits)
                )

                big_decodings, big_probs = model.inference(
                    batch,
                    logits=np.asarray(big_new_softmaxes),
                    decoder="ds",
                    top_five=False
                )
                small_decodings, small_probs = model.inference(
                    batch,
                    logits=np.asarray(small_new_softmaxes),
                    decoder="ds",
                    top_five=False
                )

                # scores increase as the token probabilities get
                # closer. this seems counter intuitive, but it works

                big_correct = big_decodings == target_phrase
                small_correct = small_decodings == target_phrase

                both = big_correct and small_correct
                big = big_correct and not small_correct
                small = not big_correct and small_correct
                neither = not big_correct and not small_correct

                if both:

                    # edge case where we haven't set initial kappa high enough
                    # so both settings work

                    argmax = "".join(
                        l_map(
                            lambda x: TOKENS[x],
                            np.argmax(
                                big_new_logits[0][:n_feats],
                                axis=1
                            )
                        )
                    )

                    result["kappa"] = small_kappa
                    result["decoding"] = big_decodings
                    result["score"] = big_probs
                    result["new_softmax"] = big_new_softmaxes[:n_feats]
                    result["new_logits"] = big_new_logits[0][:n_feats]
                    result["argmax"] = argmax

                if big_decodings != small_decodings \
                        and big_decodings == target_phrase \
                        and result["score"] < big_probs:

                    # great success!

                    best_kappa = main_kappa
                    kappa = main_kappa - 1 / current_depth
                    kappa = np.round(
                        kappa,
                        int(np.log10(max_depth))
                    )

                    # store the db_outputs
                    argmax = "".join(
                        l_map(
                            lambda x: TOKENS[x],
                            np.argmax(
                                big_new_logits[0][:n_feats],
                                axis=1
                            )
                        )
                    )

                    result["kappa"] = best_kappa
                    result["decoding"] = big_decodings
                    result["score"] = big_probs
                    result["new_softmax"] = big_new_softmaxes[:n_feats]
                    result["new_logits"] = big_new_logits[0][:n_feats]
                    result["argmax"] = argmax




                elif result["score"] != 0.0:
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
                        kappa = kappa - 1 / current_depth
                        kappa = np.round(
                            kappa,
                            int(np.log10(max_depth))
                        )

                elif kappa <= -MAX_KAPPA:
                    # we've hit a minimum boundary condition
                    break
                else:
                    # we haven't found anything yet
                    kappa = kappa - 1 / current_depth
                    kappa = np.round(
                        kappa,
                        int(np.log10(max_depth))
                    )

            if result["decoding"] != target_phrase:
                # we've not been successful, we probably won't
                # find anything useful by increasing the number of
                # repeats so break out of the loop
                break

            else:
                yield idx, result

def write_results(result, original_probs, original_decodings):
    # TODO: skip later repeats if *NOTHING* was successful

    # print out how we've done

    log("", wrap=True)

    s = "FOUND OPTIMAL KAPPA: {k}".format(
        k=result["kappa"]
    )
    s += "\tabsolute distance: {}".format(
        np.sum(result["new_logits"] - result[
            "original_logits"])
    )

    s += "\tn_feats {}".format(result["n_feats"])

    s += "\nORIG\tscore: {}".format(
        original_probs
    )
    s += "\tscore per char.: {}".format(
        original_probs / len(original_decodings)
    )
    s += "\tdecoding: {}".format(
        original_decodings
    )

    s += "\nNEW\tscore: {}".format(
        result["score"]
    )
    s += "\tscore per char.: {}".format(
        result["score"] / len(result["decoding"])
    )
    s += "\tdecoding: {}".format(
        result["decoding"]
    )

    s += "\nARGMAX: {}".format(result["argmax"])
    log(s, wrap=True)

    # finally store the db_outputs in a json file with the
    # absolute filepath *and* the original example as it
    # makes loading data for the actual optimisation attack
    # a hell of a lot easier
    if not os.path.exists(OUTDIR):
        os.mkdir(OUTDIR)

    db_path = "{b}_rep-{r}_targid-{t}".format(
        b=result["audio_basename"].rstrip(".wav"),
        r=result["repeats"],
        t=result["targ_id"]
    )

    example_db = SingleJsonDB(OUTDIR)
    example_db.open(db_path).put(result)

    s = "Wrote to file at: {}".format(db_path)
    log(s, wrap=True)


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

            write_results(
                result,
                original_probs[idx],
                original_decodings[idx],
            )


def main():

    assert MAX_DEPTH > 1.0 and MAX_DEPTH % 1 == 0

    audio_fps_etl = Batches.AllAudioFilePaths(
        INDIR,
        AUDIO_EXAMPLES_POOL,
        sort_by_file_size="desc",
        filter_term=".wav",
        max_samples=100000
    )
    all_audio_file_paths = audio_fps_etl.extract().transform().load()

    transcriptions_etl = Batches.AllTargetPhrases(
        os.path.join(INDIR, "cv-valid-test.csv"), TARGETS_POOL
    )
    all_transcriptions = transcriptions_etl.extract().transform().load()

    # Generate the batches in turn, rather than all in one go ...

    batch_factory = Generators.BatchGenerator(
        all_audio_file_paths, all_transcriptions, BATCH_SIZE,
    )

    # ... To save resources by only running the final ETLs on a batch of data

    batch_gen = batch_factory.generate(
        Batches.AudioExamples, Batches.TargetPhrases, Batches.ValidationFeeds
    )

    # Create the factory we'll use to iterate over N examples at a time.
    with AttackSpawner(gpu_device=0, max_processes=4, delay=10) as spawner:

        for (b_id, batch) in batch_gen:
            spawner.spawn(run, batch)


if __name__ == '__main__':
    main()
