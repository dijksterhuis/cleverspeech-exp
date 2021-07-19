#!/usr/bin/env python3
import os
import traceback
import tensorflow as tf
import multiprocessing as mp

from cleverspeech import data
from cleverspeech.utils.Utils import log
from cleverspeech.utils.runtime.TensorflowRuntime import TFRuntime
from cleverspeech.utils.runtime.ExperimentArguments import args

# victim model
from SecEval import VictimAPI as DeepSpeech


def custom_executor(settings, batch, model_fn):

    tf_runtime = TFRuntime(settings["gpu_device"])
    with tf_runtime.session as sess, tf_runtime.device as tf_device:

        model = model_fn(sess, batch, settings)

        decodings, probs = model.inference(
            batch,
            feed=model.feeds.examples,
        )

        logits, softmax = sess.run(
            [tf.transpose(model.raw_logits, [1, 0, 2]), model.logits],
            feed_dict=model.feeds.examples,
        )
        print(logits.shape)

        outdir = os.path.join(settings["outdir"], "validation")
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)

        for idx, basename in enumerate(batch.audios["basenames"]):
            outpath = os.path.join(outdir, basename.rstrip(".wav") + ".json")

            res = {
                "decoding": decodings[idx],
                "probs": probs[idx],
                "logits": logits[idx],
                "softmax": softmax[idx],
            }

            json_res = data.egress.load.prepare_json_data(res)

            with open(outpath, "w+") as f:
                f.write(json_res)

            log("Ran validation for example {}".format(idx), wrap=False)


def custom_manager(settings, attack_fn, batch_gen):

    for b_id, batch in batch_gen:

        attack_process = mp.Process(
            target=custom_executor,
            args=(settings, batch, attack_fn)
        )

        try:
            attack_process.start()
            attack_process.join()

        except Exception as e:
            tb = "".join(traceback.format_exception(None, e, e.__traceback__))

            s = "Something broke! Attack failed to run for these examples:\n"
            s += '\n'.join(batch.audios["basenames"])
            s += "\n\nError Traceback:\n{e}".format(e=tb)

            log(s, wrap=True)
            attack_process.terminate()
            raise


def create_validation_graph(sess, batch, settings):

    feeds = data.ingress.Feeds.Validation(batch)

    audios_ph = tf.placeholder(tf.float32, [batch.size, batch.audios["max_samples"]], name="new_input")
    audio_lengths_ph = tf.placeholder(tf.int32, [batch.size], name='qq_featlens')

    feeds.create_feeds(audios_ph, audio_lengths_ph)

    model = DeepSpeech.Model(
        sess, audios_ph, batch,
        decoder=settings["decoder"],
        beam_width=settings["beam_width"]
    )

    # hacky
    model.feeds = feeds

    return model


def attack_run(master_settings):

    batch_gen = data.ingress.etl.batch_generators.standard(master_settings)

    custom_manager(
        master_settings,
        create_validation_graph,
        batch_gen,
    )

    log("Finished all runs.")


if __name__ == '__main__':

    log("", wrap=True)

    args(attack_run)



