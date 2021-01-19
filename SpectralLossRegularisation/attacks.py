#!/usr/bin/env python3
import os

# attack def imports
from cleverspeech.graph.GraphConstructor import Constructor
from cleverspeech.graph import Constraints
from cleverspeech.graph import Graphs
from cleverspeech.graph.Losses import CTCLoss
from cleverspeech.graph import Optimisers
from cleverspeech.graph import Procedures
from cleverspeech.graph import Outputs

from SecEval import VictimAPI as DeepSpeech


# boilerplate imports
from cleverspeech.data import ETL
from cleverspeech.data import Feeds
from cleverspeech.data import Generators
from cleverspeech.utils.Utils import log, l_map, args

from boilerplate import execute

import Losses

GPU_DEVICE = 0
MAX_PROCESSES = 4
SPAWN_DELAY = 60 * 5

TOKENS = " abcdefghijklmnopqrstuvwxyz'-"
BEAM_WIDTH = 500

AUDIOS_INDIR = "/data/samples/all/"
TARGETS_PATH = "/data/samples/cv-valid-test.csv"
OUTDIR = "/data/adv/baseline/"
MAX_EXAMPLES = 1000
MAX_TARGETS = 1000
MAX_AUDIO_LENGTH = 120000

RESCALE = 0.95
CONSTRAINT_UPDATE = "geom"
LEARNING_RATE = 10
NUMB_STEPS = 10000
DECODING_STEP = 10
BATCH_SIZE = 10

N_RUNS = 5


def get_batch_generator(settings):

    # get N samples of all the data. alsp make sure to limit example length,
    # otherwise we'd have to do adaptive batch sizes.

    audio_etl = ETL.AllAudioFilePaths(
        settings["audio_indir"],
        MAX_EXAMPLES,
        filter_term=".wav",
        max_samples=MAX_AUDIO_LENGTH
    )

    all_audio_file_paths = audio_etl.extract().transform().load()

    targets_etl = ETL.AllTargetPhrases(
        settings["targets_path"], MAX_TARGETS,
    )
    all_targets = targets_etl.extract().transform().load()

    # hack the targets data for the naive non-merging CTC experiment

    if "n_repeats" in settings.keys():
        all_targets = l_map(
            lambda x: "".join([i * settings["n_repeats"] for i in x]),
            all_targets
        )

    # Generate the batches in turn, rather than all in one go ...

    batch_factory = Generators.BatchGenerator(
        all_audio_file_paths, all_targets, settings["batch_size"]
    )

    # ... To save resources by only running the final ETLs on a batch of data

    batch_gen = batch_factory.generate(
        ETL.AudioExamples, ETL.TargetPhrases, Feeds.Attack
    )

    log(
        "New Run",
        "Number of test examples: {}".format(batch_factory.numb_examples),
        ''.join(["{k}: {v}\n".format(k=k, v=v) for k, v in settings.items()]),
    )
    return batch_gen


def spectral_run(master_settings):
    """
    CTC Loss attack modified from the original Carlini & Wagner work.

    Using a hard constraint is better for security evaluations, so we ignore the
    L2 distance regularisation term in the optimisation goal.

    TODO: I could probably remove `Base.add_distance_loss()` method...?

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

        attack.add_adversarial_loss(CTCLoss)
        attack.add_distance_loss(Losses.SpectralLoss)
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

    outdir = os.path.join(OUTDIR, "spectral/")

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
    }

    settings.update(master_settings)
    batch_gen = get_batch_generator(settings)

    execute(settings, create_attack_graph, batch_gen)

    log("Finished run.")  # {}.".format(run))


if __name__ == '__main__':

    log("", wrap=True)

    experiments = {
        "baseline": spectral_run,
    }

    args(experiments)



