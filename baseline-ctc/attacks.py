#!/usr/bin/env python3
import os
import traceback
import multiprocessing as mp

from cleverspeech import data
from cleverspeech import graph
from cleverspeech.utils.runtime.ExperimentArguments import args

from cleverspeech.utils.Utils import log

# victim model
from SecEval import VictimAPI as DeepSpeech


def local_writer_fn(queue, settings):

    import traceback
    from cleverspeech.data.egress import transform, load

    # set up the directory we'll use for local results data

    if not os.path.exists(settings["outdir"]):
        os.makedirs(settings["outdir"], exist_ok=True)
        log("Created new directory: {}".format(settings["outdir"]))

    # Write the current settings to "settings.json" file.

    settings_db = load.LocalJsonMetadataFile(settings["outdir"])
    settings_db.open("settings")
    settings_db.put(settings)
    log("Wrote settings.")

    json_db = load.LocalJsonMetadataFile(settings["outdir"])
    wav_db = load.LocalWavFiles(settings["outdir"])

    while True:

        results = queue.get()

        if results == "dead":

            # This is a signal telling us there's no more data inbound.
            # Obviously we don't want to write this to disk, so tell parent
            # that we're done with this item and break the while loop.

            queue.task_done()
            break

        elif data != "dead":
            try:
                for example_data in transform.evasion_transforms(results):
                    log(
                        transform.evasion_logging(example_data),
                        wrap=False,
                        outdir=settings["outdir"],
                        stdout=False,
                        timings=True,
                    )
                    if example_data["success"] is True:

                        db_file_path = example_data['basenames'].rstrip(".wav")
                        json_db.open(db_file_path)
                        json_db.put(example_data)

                        # -- Write audio data.

                        for wav_file in ["audio", "deltas", "advs"]:

                            wav_db.open(db_file_path + "_{}".format(wav_file))
                            wav_db.put(example_data[wav_file])

            except Exception as e:
                tb = "".join(
                    traceback.format_exception(None, e, e.__traceback__))

                s = "Something broke during file writes!"
                s += "\n\nError Traceback:\n{e}".format(e=tb)
                log(s, wrap=True)
                raise

            finally:
                queue.task_done()


def s3_writer_fn(queue, settings):

    import os
    import traceback
    from cleverspeech.data.egress import transform, load

    # set up the directory we'll use for local logging data

    if not os.path.exists(settings["outdir"]):
        os.makedirs(settings["outdir"], exist_ok=True)
        log("Created new directory: {}".format(settings["outdir"]))

    # Write the current settings to "settings.json" file.

    settings_db = load.S3JsonMetadataFile("cleverspeech-results")
    settings_db.open(os.path.join(settings["outdir"], "settings"))
    settings_db.put(settings)
    log("Wrote settings.")

    metadata_db = load.S3JsonMetadataFile("cleverspeech-results")

    while True:

        results = queue.get()

        if results == "dead":

            # This is a signal telling us there's no more data inbound.
            # Obviously we don't want to write this to disk, so tell parent
            # that we're done with this item and break the while loop.

            queue.task_done()
            break

        elif data != "dead":
            try:
                for example_data in transform.evasion_transforms(results):
                    log(
                        transform.evasion_logging(example_data),
                        wrap=False,
                        outdir=settings["outdir"],
                        stdout=False,
                        timings=True,
                    )
                    if example_data["success"] is True:

                        file_path = example_data['basenames'].rstrip(".wav")

                        # S3 writes write out the metadata for *all* examples
                        # not just the latest one

                        file_path = os.path.join(
                            file_path,
                            "eps_{}".format(example_data["bounds_eps"])
                        )

                        metadata_db.open(
                            os.path.join(settings["outdir"], file_path)
                        )
                        metadata_db.put(example_data)

            except Exception as e:
                tb = "".join(
                    traceback.format_exception(None, e, e.__traceback__))

                s = "Something broke during file writes!"
                s += "\n\nError Traceback:\n{e}".format(e=tb)
                log(s, wrap=True)
                raise

            finally:
                queue.task_done()


def executor(results_queue, settings, batch, attack_fn):
    from cleverspeech.data.egress import extract
    from cleverspeech.utils.runtime.TensorflowRuntime import TFRuntime
    from cleverspeech.utils.Utils import log

    # tensorflow sessions can't be passed between processes
    tf_runtime = TFRuntime(settings["gpu_device"])

    with tf_runtime.session as sess, tf_runtime.device as tf_device:

        # Initialise attack graph constructor function
        attack = attack_fn(sess, batch, settings)

        # log some useful things for debugging before the attack runs
        attack.validate()

        s = "Created Attack Graph and Feeds. Loaded TF Operations:"
        log(s, wrap=False)
        log(funcs=tf_runtime.log_attack_tensors)

        s = "Beginning attack run...\nMonitor progress in: {}".format(
            settings["outdir"] + "log.txt"
        )
        log(s)

        for is_results_step in attack.run():
            if is_results_step:
                res = extract.get_evasion_attack_state(attack)
                results_queue.put(res)


def manager(settings, attack_fn, batch_gen):

    results_queue = mp.JoinableQueue()

    if settings["writer"] == "local":
        writer_process = mp.Process(
            target=local_writer_fn,
            args=(results_queue, settings)
        )
    elif settings["writer"] == "s3":
        writer_process = mp.Process(
            target=s3_writer_fn,
            args=(results_queue, settings)
        )
    else:
        raise NotImplementedError

    writer_process.start()
    log("Started a writer subprocess.")

    for b_id, batch in batch_gen:

        # we *must* call the tensorflow session within the batch loop so the
        # graph gets reset: the maximum example length in a batch affects the
        # size of most graph elements.

        log("Running for Batch Number: {}".format(b_id), wrap=True)

        attack_process = mp.Process(
            target=executor,
            args=(results_queue, settings, batch, attack_fn)
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
            raise

        finally:

            log("Attempting to close writer queue and subprocess.", wrap=True)

            attack_process.terminate()
            log("Attack subprocess closed.", wrap=True)

            results_queue.put("dead")
            results_queue.close()
            log("Results queue closed.", wrap=True)

            writer_process.join()
            writer_process.terminate()
            log("Writer subprocess closed.", wrap=True)


# ==============================================================================


LOSS_CHOICES = {
    "ctc": graph.Losses.CTCLoss,
    "ctc2": graph.Losses.CTCLossV2,
}


def create_attack_graph(sess, batch, settings):

    feeds = data.ingress.Feeds.Attack(batch)

    attack = graph.AttackConstructors.EvasionAttackConstructor(sess, batch, feeds)
    attack.add_placeholders(graph.Placeholders.Placeholders)
    attack.add_hard_constraint(
        graph.Constraints.L2,
        r_constant=settings["rescale"],
        update_method=settings["constraint_update"],
    )
    attack.add_perturbation_subgraph(
        graph.PerturbationSubGraphs.Independent
    )
    attack.add_victim(
        DeepSpeech.Model,
        decoder=settings["decoder"],
        beam_width=settings["beam_width"]
    )
    attack.add_loss(
        LOSS_CHOICES[settings["loss"]]
    )
    attack.create_loss_fn()
    attack.add_optimiser(
        graph.Optimisers.AdamIndependentOptimiser,
        learning_rate=settings["learning_rate"]
    )
    attack.add_procedure(
        graph.Procedures.StandardProcedure,
        steps=settings["nsteps"],
        update_step=settings["decode_step"]
    )

    return attack


def attack_run(master_settings):
    """
    CTC Loss attack modified from the original Carlini & Wagner work.
    """

    loss = master_settings["loss"]
    decoder = master_settings["decoder"]
    outdir = master_settings["outdir"]

    outdir = os.path.join(outdir, "evasion/baselines/ctc/")
    outdir = os.path.join(outdir, "{}/".format(loss))
    outdir = os.path.join(outdir, "{}/".format(decoder))

    master_settings["outdir"] = outdir

    batch_gen = data.ingress.etl.batch_generators.standard(master_settings)
    manager(master_settings, create_attack_graph, batch_gen)

    log("Finished run.")


if __name__ == '__main__':

    log("", wrap=True)

    extra_args = {
        "loss": [str, "ctc", False, LOSS_CHOICES.keys()],
    }

    args(attack_run, additional_args=extra_args)

