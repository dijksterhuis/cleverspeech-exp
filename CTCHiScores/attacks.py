#!/usr/bin/env python3
import os

from cleverspeech.graph.GraphConstructor import Constructor
from cleverspeech.graph import Constraints
from cleverspeech.graph import Graphs
from cleverspeech.graph import Optimisers
from cleverspeech.graph import Procedures
from cleverspeech.graph import Outputs
from cleverspeech.data import Feeds
from cleverspeech.data import ETL
from cleverspeech.utils.Utils import log, args, l_map

from SecEval import VictimAPI as DeepSpeech


from boilerplate import execute

# local attack classes
import custom_defs
import custom_etl

GPU_DEVICE = 0
MAX_PROCESSES = 3
SPAWN_DELAY = 60 * 5

TOKENS = " abcdefghijklmnopqrstuvwxyz'-"
BEAM_WIDTH = 500

AUDIOS_INDIR = "./samples/all/"
TARGETS_PATH = "./samples/cv-valid-test.csv"
OUTDIR = "./adv/hi-scores/"

# targets search parameters
MAX_EXAMPLES = 1000
MAX_TARGETS = 1000
MAX_AUDIO_LENGTH = 120000

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


def get_batch_factory(settings):

    # get N samples of all the data. alsp make sure to limit example length,
    # otherwise we'd have to do adaptive batch sizes.

    audio_etl = ETL.AllAudioFilePaths(
        settings["audio_indir"],
        settings["max_examples"],
        filter_term=".wav",
        max_samples=settings["max_audio_length"]
    )

    all_audio_file_paths = audio_etl.extract().transform().load()

    targets_etl = ETL.AllTargetPhrases(
        settings["targets_path"], settings["max_targets"],
    )
    all_targets = targets_etl.extract().transform().load()

    # hack the targets data for the naive non-merging CTC experiment

    if "n_repeats" in settings.keys():
        all_targets = l_map(
            lambda x: "".join([i * settings["n_repeats"] for i in x]),
            all_targets
        )

    # Generate the batches in turn, rather than all in one go ...

    batch_factory = custom_etl.CTCHiScoresBatchGenerator(
        all_audio_file_paths, all_targets, settings["batch_size"]
    )

    # ... To save resources by only running the final ETLs on a batch of data

    batch_gen = batch_factory.generate(
        ETL.AudioExamples, custom_etl.RepeatsTargetPhrases, Feeds.Attack
    )

    log(
        "New Run",
        "Number of test examples: {}".format(batch_factory.numb_examples),
        ''.join(["{k}: {v}\n".format(k=k, v=v) for k, v in settings.items()]),
    )
    return batch_gen


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

        attack.add_outputs(
            Outputs.Base,
            settings["outdir"],
        )

        return attack

    for run in range(0, N_RUNS):

        outdir = os.path.join(OUTDIR, "ctc-repeats/")
        outdir = os.path.join(outdir, "{}/".format(LOGITS_SEARCH))
        outdir = os.path.join(outdir, "run_{}/".format(run))

        settings = {
            "audio_indir": AUDIOS_INDIR,
            "targets_path": TARGETS_PATH,
            "outdir": outdir,
            "batch_size": BATCH_SIZE,
            "tokens": TOKENS,
            "nsteps": NUMB_STEPS,
            "decode_step": DECODING_STEP,
            "beam_width": BEAM_WIDTH,
            "constraint_update": CONSTRAINT_UPDATE,
            "rescale": RESCALE,
            "learning_rate": LEARNING_RATE,
            "gpu_device": GPU_DEVICE,
            "max_spawns": MAX_PROCESSES,
            "spawn_delay": SPAWN_DELAY,
            "max_examples": MAX_EXAMPLES,
            "max_targets": MAX_TARGETS,
            "max_audio_length": MAX_AUDIO_LENGTH,
        }

        settings.update(master_settings)

        batch_factory = get_batch_factory(settings)

        execute(settings, create_attack_graph, batch_factory)

        log("Finished run {}.".format(run))


if __name__ == '__main__':

    experiments = {
        "ctcrepeats": ctc_repeats_run,
    }

    args(experiments)

