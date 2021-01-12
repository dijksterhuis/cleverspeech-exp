#!/usr/bin/env python3
import os

from cleverspeech.Attacks.Base import Constructor
from cleverspeech.Attacks import Constraints
from cleverspeech.Attacks import Graphs
#from cleverspeech.Attacks import Losses
from cleverspeech.Attacks import Optimisers
from cleverspeech.Attacks import Procedures

from cleverspeech.Models import DeepSpeech
from cleverspeech.Data import ETL
from cleverspeech.Data import Feeds
from cleverspeech.Data import Generators

from cleverspeech.Utils import log, args

from boilerplate import execute

# local attack classes
import custom_defs

GPU_DEVICE = 0
MAX_PROCESSES = 3
SPAWN_DELAY = 60 * 5

TOKENS = " abcdefghijklmnopqrstuvwxyz'-"
BEAM_WIDTH = 500

INDIR = "/data/target-logits/"
OUTDIR = "/data/adv/hi-scores/"

# targets search parameters
MAX_EXAMPLES = 1000

RESCALE = 0.95
CONSTRAINT_UPDATE = "geom"
LEARNING_RATE = 10
NUMB_STEPS = 10000
DECODING_STEP = 10
BATCH_SIZE = 10
LOGITS_SEARCH = "sbgs"

N_RUNS = 1


"""
Tips on new attack definitions:

0. Prefer to make changes to in ./experiments first, instead of jumping straight 
into changing ./cleverspeech/Attacks  

1. Get `create_attack_graph` function working first, then worry about settings 
-- you can always perform hyperparameter searches.

2. Do you need to think a bit harder about how the rest of the code structure? 
Don't be afraid to change things -- it's all a learning process!

3. Keep everything in a settings dictionary to explicitly document exact 
parameters used for experiments.

4.  If something with attack classes / objects needs to change, don't hack it.
Think about the change and refactor anything that needs refactoring. You'll
thank me in the long run.

5. Procedures are a bit f**ked at the moment. I need to have a think on this.  

"""


def get_logits_batch_generator(settings):
    # get a N samples of all the data

    target_logits_etl = ETL.AllTargetLogitsAndAudioFilePaths(
        settings["indir"], MAX_EXAMPLES,
    )
    all_data = target_logits_etl.extract().transform().load()

    # Generate the batches in turn, rather than all in one go ...

    batch_factory = Generators.LogitsBatchGenerator(
        all_data, settings["batch_size"]
    )

    # ... To save resources by only running the final ETLs on a batch of data

    batch_gen = batch_factory.generate(
        ETL.AudioExamples, ETL.TargetLogits, Feeds.HiScores
    )

    log(
        "New Run",
        "Number of test examples: {}".format(batch_factory.numb_examples),
        ''.join(["{k}: {v}\n".format(k=k, v=v) for k, v in settings.items()]),
    )
    return batch_gen


def get_batch_factory(settings):
    # get a N samples of all the data

    target_logits_etl = ETL.AllTargetLogitsAndAudioFilePaths(
        settings["indir"], MAX_EXAMPLES, filter_term="latest"
    )
    all_data = target_logits_etl.extract().transform().load()

    # Generate the batches in turn, rather than all in one go ...
    # ... To save resources by only running the final ETLs on a batch of data

    batch_factory = Generators.LogitsBatchGenerator(
        all_data, settings["batch_size"]
    )

    log(
        "New Run",
        "Number of test examples: {}".format(batch_factory.numb_examples),
        ''.join(["{k}: {v}\n".format(k=k, v=v) for k, v in settings.items()]),
    )
    return batch_factory


def ctc_repeats_run(master_settings):
    """
    Example to show that just ignoring repeats during CTC loss pre-processing
    steps doesn't help us -- we get a lot of repeated characters in the final
    transcription.

    The below settings are an edge case for CTC Loss use and are enabled in the
    RepeatsCTCLoss object:
    ```
        preprocess_collapse_repeated=False,
        ctc_merge_repeated=True,
    ```

    :return: None
    """
    def create_attack_graph(sess, batch, settings):

        attack = Constructor(sess, batch)

        attack.add_hard_constraint(
            Constraints.L2,
            r_constant=settings["rescale"],
            update_method=settings["constraint_update"],
        )

        attack.add_graph(
            Graphs.SimpleAttack
        )

        attack.add_victim(
            DeepSpeech.Model,
            tokens=settings["tokens"],
            beam_width=settings["beam_width"]
        )

        attack.add_adversarial_loss(custom_defs.RepeatsCTCLoss)

        attack.create_loss_fn()

        attack.add_optimiser(
            Optimisers.AdamOptimiser,
            learning_rate=settings["learning_rate"]
        )

        attack.add_procedure(
            Procedures.UpdateBound,
            steps=settings["nsteps"],
            decode_step=settings["decode_step"]
        )

        return attack

    for run in range(0, N_RUNS):

        outdir = os.path.join(OUTDIR, "ctc-repeats/")
        outdir = os.path.join(outdir, "{}/".format(LOGITS_SEARCH))
        outdir = os.path.join(outdir, "run_{}/".format(run))

        settings = {
            "indir": os.path.join(INDIR, "{}/".format(LOGITS_SEARCH)),
            "outdir": outdir,
            "batch_size": BATCH_SIZE,
            "tokens": TOKENS,
            "nsteps": NUMB_STEPS,
            "decode_step": DECODING_STEP,
            "beam_width": BEAM_WIDTH,
            "constraint_update": CONSTRAINT_UPDATE,
            "rescale": RESCALE,
            "learning_rate": LEARNING_RATE,
            "logits_search": LOGITS_SEARCH,
            "gpu_device": GPU_DEVICE,
            "max_spawns": MAX_PROCESSES,
            "spawn_delay": SPAWN_DELAY,
            "n_repeats": 3,
        }

        settings.update(master_settings)

        batch_factory = get_batch_factory(settings)
        batch_gen = batch_factory.generate(
            ETL.AudioExamples, ETL.TargetLogits, Feeds.CTCRepeats
        )

        execute(settings, create_attack_graph, batch_gen)

        log("Finished run {}.".format(run))


if __name__ == '__main__':

    experiments = {
        "ctcrepeats": ctc_repeats_run,
    }

    args(experiments)

