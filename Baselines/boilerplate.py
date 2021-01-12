import os
import sys
import argparse


from cleverspeech.Data.Results import SingleJsonDB, step_logging, success_logging
from cleverspeech.Evaluation import Processing as BasicProcessing
from cleverspeech.RuntimeUtils import create_tf_runtime, log_attack_tensors, AttackSpawner
from cleverspeech.Utils import log, dump_wavs, run_decoding_check


def execute(settings, attack_fn, batch_gen):

    # set up the directory we'll use for results

    if not os.path.exists(settings["outdir"]):
        os.makedirs(settings["outdir"], exist_ok=True)

    # Write the current settings to "settings.json" file.

    settings_db = SingleJsonDB(settings["outdir"])
    settings_db.open("settings").put(settings)
    log("Wrote settings.")

    # Manage GPU memory and CPU processes usage.

    attack_spawner = AttackSpawner(
        gpu_device=settings["gpu_device"],
        max_processes=settings["max_spawns"],
        delay=settings["spawn_delay"],
    )

    with attack_spawner as spawner:
        for b_id, batch in batch_gen:
            spawner.spawn(boilerplate, settings, attack_fn, batch)

    # Run the standard stats script on all successful examples once all attacks
    # are completed.

    BasicProcessing.batch_generate_statistic_file(settings["outdir"])


def boilerplate(settings, attack_fn, batch):

    # we *must* call the tensorflow session within the batch loop so the
    # graph gets reset: the maximum example length in a batch affects the
    # size of most graph elements.

    # tensorflow sessions can't be passed between processes either, so we have
    # to create it here.

    tf_session, tf_device = create_tf_runtime(settings["gpu_device"])
    with tf_session as sess, tf_device:

        # Initialise curried attack graph constructor function

        attack = attack_fn(sess, batch, settings)

        # create placeholder feeds

        batch.feeds.create_feeds(attack.graph)

        # log some useful things for debugging before the attack runs

        run_decoding_check(attack, batch)
        log("Created Attack Graph and Feeds.")

        log("Loaded TF Operations:")
        log(funcs=log_attack_tensors)

        # Run the attack generator loop. See `Attacks/Procedures.py` for
        # detailed info on returned results.
        log("Beginning attack run...\nMonitor progress in: {}".format(
            settings["outdir"] + "log.txt"
        ))

        for idx, step_result, example in attack.run():

            # log how we're doing to file on disk
            # if you want to monitor progress live then you'll have to do:
            # `watch -n 1 tail -n 20 ./path/to/out/dir/log.txt`.
            step_logs = step_logging(step_result)
            log(
                step_logs,
                wrap=False,
                outdir=settings["outdir"],
                stdout=False
            )

            if example["success"] is True:

                # log success to stdout as that's nice to know about.
                log(
                    success_logging(example), wrap=False
                )

                # Write most recent results to a json file.
                # note that SingleJSONDB overwrites for each new example.

                example_db = SingleJsonDB(settings["outdir"])
                example_db.open(
                    example['basename'].rstrip(".wav")
                ).put(example)

                # Store most recently successful audio data.

                dump_wavs(
                    settings["outdir"],
                    example,
                    ["original", "delta", "advex"],
                    filepath_key="basename",
                    sample_rate=16000
                )
