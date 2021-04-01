#!/usr/bin/env python3
import os

# attack def imports
from cleverspeech.graph.GraphConstructor import Constructor
from cleverspeech.graph import Constraints
from cleverspeech.graph import VariableGraphs
from cleverspeech.graph import Losses
from cleverspeech.graph import Optimisers
from cleverspeech.graph import Procedures

from cleverspeech.data.ingress.etl import batch_generators
from cleverspeech.data.ingress import Feeds
from cleverspeech.data.egress.Databases import SingleJsonDB
from cleverspeech.data.egress.Transforms import Standard
from cleverspeech.data.egress.Writers import SingleFileWriter
from cleverspeech.data.egress import Reporting

from cleverspeech.utils.Utils import log
from cleverspeech.utils.runtime.AttackSpawner import AttackSpawner
from cleverspeech.utils.runtime.ExperimentArguments import args

# victim model
from SecEval import VictimAPI as Victim

GPU_DEVICE = 0
MAX_PROCESSES = 1
SPAWN_DELAY = 30

AUDIOS_INDIR = "./samples/all/"
TARGETS_PATH = "./samples/cv-valid-test.csv"
OUTDIR = "./adv/baselines/ctc"
MAX_EXAMPLES = 100
MAX_TARGETS = 1000
MAX_AUDIO_LENGTH = 120000

TOKENS = " abcdefghijklmnopqrstuvwxyz'-"
BEAM_WIDTH = 500
LEARNING_RATE = 10
CONSTRAINT_UPDATE = "geom"
RESCALE = 0.95
DECODING_STEP = 100
NUMB_STEPS = DECODING_STEP ** 2
BATCH_SIZE = 10

# extreme run settings
LOSS_UPDATE_THRESHOLD = 10.0
KAPPA = 5.0

LOSSES = {
    "ctc": Losses.CTCLoss,
    "ctc_v2": Losses.CTCLossV2,
}


def execute(settings, attack_fn, batch_gen):

    # set up the directory we'll use for results

    if not os.path.exists(settings["outdir"]):
        os.makedirs(settings["outdir"], exist_ok=True)

    results_extracter = Standard()
    file_writer = SingleFileWriter(settings["outdir"], results_extracter)

    # Write the current settings to "settings.json" file.

    settings_db = SingleJsonDB(settings["outdir"])
    settings_db.open("settings").put(settings)
    log("Wrote settings.")

    # Manage GPU memory and CPU processes usage.

    attack_spawner = AttackSpawner(
        gpu_device=settings["gpu_device"],
        max_processes=settings["max_spawns"],
        delay=settings["spawn_delay"],
        file_writer=file_writer,
    )

    with attack_spawner as spawner:
        for b_id, batch in batch_gen:
            log("Running for Batch Number: {}".format(b_id), wrap=True)
            spawner.spawn(settings, attack_fn, batch)

    # Run the stats function on all successful examples once all attacks
    # are completed.
    Reporting.generate_stats_file(settings["outdir"])


def create_attack_graph(sess, batch, settings):

    feeds = Feeds.Attack(batch)

    attack = Constructor(sess, batch, feeds)

    attack.add_hard_constraint(
        Constraints.L2,
        r_constant=settings["rescale"],
        update_method=settings["constraint_update"],
    )

    attack.add_graph(
        VariableGraphs.Independent
    )

    attack.add_victim(
        Victim.Model,
        tokens=settings["tokens"],
        beam_width=settings["beam_width"]
    )

    attack.add_loss(
        LOSSES[settings["loss"]]
    )
    attack.create_loss_fn()

    attack.add_optimiser(
        Optimisers.AdamIndependentOptimiser,
        learning_rate=settings["learning_rate"]
    )

    attack.add_procedure(
        Procedures.UpdateOnDecoding,
        steps=settings["nsteps"],
        decode_step=settings["decode_step"]
    )

    attack.create_feeds()

    return attack


def ctc_run(master_settings):
    """
    CTC Loss attack modified from the original Carlini & Wagner work.
    """

    loss = "ctc"
    outdir = os.path.join(OUTDIR, "{}/".format(loss))

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
        "loss": loss,
    }

    settings.update(master_settings)
    batch_gen = batch_generators.standard(settings)
    execute(settings, create_attack_graph, batch_gen)
    log("Finished run.")  # {}.".format(run))


def ctc_v2_run(master_settings):
    """
    CTC Loss attack modified from the original Carlini & Wagner work.
    """

    loss = "ctc_v2"
    outdir = os.path.join(OUTDIR, "{}/".format(loss))

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
        "loss": loss,
    }

    settings.update(master_settings)
    batch_gen = batch_generators.standard(settings)
    execute(settings, create_attack_graph, batch_gen)
    log("Finished run.")  # {}.".format(run))


if __name__ == '__main__':

    log("", wrap=True)
    experiments = {"ctc": ctc_run, "ctc2": ctc_v2_run}
    args(experiments)



