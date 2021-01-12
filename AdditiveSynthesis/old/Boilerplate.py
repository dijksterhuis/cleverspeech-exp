import os
#import pynvml

from cleverspeech.Data import Batches
from cleverspeech.Data import Generators
from cleverspeech.Data.Results import SingleJsonDB, step_logging
from cleverspeech.Evaluation import Processing
from cleverspeech.RuntimeUtils import create_tf_runtime, log_attack_tensors
from cleverspeech.Utils import log, dump_wavs


# def get_gpu_memory_usage(gpu_idx=0):
#     pynvml.nvml.nvmlInit()
#     handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
#     mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#     return mem_info.free, mem_info.used, mem_info.total


def run_decoding_check(attack, batch, beam_width=500):
    """
    Do an initial decoding to verify everything is working
    """
    decodings, probs = attack.victim.inference(
        batch,
        feed=batch.feeds.examples,
        decoder="batch"
    )
    log("Initial decodings:", '\n'.join([" ".join([str(p), d]) for p, d in zip(probs, decodings)]))


def run(settings):

    attack_fn = settings['fn'].pop("attack")
    synthesiser_fn = settings['fn'].pop("synth")

    # set up the directory we'll use for results
    if not os.path.exists(settings["outdir"]):
        os.makedirs(settings["outdir"], exist_ok=True)

    # Write the current settings to "settings.json" file.
    settings_db = SingleJsonDB(settings["outdir"])
    settings_db.open("settings").put(settings)
    log("Wrote settings.")

    # Create the factory we'll use to iterate over N examples at a time.
    batch_factory = Generators.BatchGenerator(
        settings["indir"],
        settings["outdir"],
        settings["target"],
        tokens=settings["tokens"],
        sort_by_file_size="desc",
    )

    # Create a generator from the data factory.
    batch_gen = batch_factory.generate(Batches.AttackLoader, batch_size=settings["batch_size"])

    log(
        "New Run", "Number of test examples: {}".format(batch_factory.numb_examples),
        ''.join(["{k}: {v}\n".format(k=k, v=v) for k, v in settings.items()]),
    )

    for b_id, batch in batch_gen:

        log(funcs=batch.__str__)

        # we *must* call the tensorflow session within the batch loop so the
        # graph gets reset: the maximum example length in a batch affects the
        # size of most graph elements.

        tf_session, tf_device = create_tf_runtime(settings["gpu_device"])
        with tf_session as sess, tf_device:

            # Initialise curried synthesiser class
            synth = synthesiser_fn(batch, **settings["synth"])

            # Initialise curried attack graph constructor function
            attack = attack_fn(sess, batch, synth, settings)

            # create placeholder feeds
            batch.feeds.create_feeds(attack.graph)

            run_decoding_check(attack, batch, beam_width=settings["beam_width"])
            log("Created Synth, Attack Graph and Feeds.")

            log("Loaded TF Operations:")
            log(funcs=log_attack_tensors)


            # Run the attack generator loop. See `Attacks/Procedures.py` for
            # detailed info on returned results.
            for idx, step_result, example in attack.run():

                if idx == 0: print("*" * 30)

                # print how we're doing to stdout.
                log(step_logging(step_result), wrap=False, outdir=settings["outdir"])

                if example["success"] is True:

                    s = "Found new example: {f}".format(f=example['basename'])
                    log(s, wrap=False)

                    # Write most recent results to a json file.
                    example_db = SingleJsonDB(settings["outdir"])
                    example_db.open(example['basename'].rstrip(".wav")).put(example)

                    # Store most recently successful audio data.
                    dump_wavs(
                        settings["outdir"],
                        example,
                        ["original", "delta", "advex"],
                        filepath_key="basename",
                        sample_rate=16000
                    )

    Processing.batch_generate_statistic_file(settings["outdir"])
