#!/usr/bin/env python3
import os

# attack def imports
from cleverspeech.graph.GraphConstructor import Constructor
from cleverspeech.graph import Constraints
from cleverspeech.graph import Graphs
from cleverspeech.graph import Losses
from cleverspeech.graph import Optimisers
from cleverspeech.graph import Procedures
from cleverspeech.graph import Outputs
from cleverspeech.data import Feeds

from cleverspeech.data.etl.batch_generators import get_standard_batch_generator
from cleverspeech.data.Results import SingleFileWriter, SingleJsonDB
from cleverspeech.eval import PerceptualStatsBatch
from cleverspeech.utils.Utils import log, args
from cleverspeech.utils.RuntimeUtils import AttackSpawner

# victim model
from SecEval import VictimAPI as Victim

# custom experimental extensions
import custom_defs

GPU_DEVICE = 0
MAX_PROCESSES = 1
SPAWN_DELAY = 30

AUDIOS_INDIR = "./samples/all/"
TARGETS_PATH = "./samples/cv-valid-test.csv"
OUTDIR = "./adv/baselines/"
MAX_EXAMPLES = 100
MAX_TARGETS = 1000
MAX_AUDIO_LENGTH = 120000

TOKENS = " abcdefghijklmnopqrstuvwxyz'-"
BEAM_WIDTH = 500
LEARNING_RATE = 10
CONSTRAINT_UPDATE = "geom"
RESCALE = 0.95
DECODING_STEP = 500
NUMB_STEPS = DECODING_STEP ** 2
BATCH_SIZE = 10

# extreme run settings
LOSS_UPDATE_THRESHOLD = 10.0
KAPPA = 5.0


def execute(settings, attack_fn, batch_gen):

    # set up the directory we'll use for results

    if not os.path.exists(settings["outdir"]):
        os.makedirs(settings["outdir"], exist_ok=True)

    file_writer = SingleFileWriter(settings["outdir"])

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
    PerceptualStatsBatch.batch_generate_statistic_file(settings["outdir"])


def baseline_ctc_run(master_settings):
    """
    CTC Loss attack modified from the original Carlini & Wagner work.

    Using a hard constraint is better for security evaluations, so we ignore the
    L2 distance regularisation term in the optimisation goal.

    TODO: I could probably remove `Base.add_loss()` method...?

    :return: None
    """
    def create_attack_graph(sess, batch, settings):

        feeds = Feeds.Attack(batch)

        attack = Constructor(sess, batch, feeds)

        attack.add_hard_constraint(
            Constraints.L2,
            r_constant=settings["rescale"],
            update_method=settings["constraint_update"],
        )

        attack.add_graph(
            Graphs.SimpleAttack
        )

        attack.add_victim(
            Victim.Model,
            tokens=settings["tokens"],
            beam_width=settings["beam_width"]
        )

        attack.add_loss(Losses.CTCLoss)
        attack.create_loss_fn()

        attack.add_optimiser(
            Optimisers.AdamOptimiser,
            learning_rate=settings["learning_rate"]
        )

        attack.add_procedure(
            Procedures.UpdateOnDecoding,
            steps=settings["nsteps"],
            decode_step=settings["decode_step"]
        )

        attack.add_outputs(
            Outputs.Base,
            settings["outdir"],
        )

        attack.create_feeds()

        return attack

    outdir = os.path.join(OUTDIR, "ctc/")

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

    batch_gen = get_standard_batch_generator(settings)

    execute(settings, create_attack_graph, batch_gen)

    log("Finished run.")  # {}.".format(run))


def baseline_ctc_v2_run(master_settings):
    """
    CTC Loss attack modified from the original Carlini & Wagner work.

    Using a hard constraint is better for security evaluations, so we ignore the
    L2 distance regularisation term in the optimisation goal.

    TODO: I could probably remove `Base.add_loss()` method...?

    :return: None
    """
    def create_attack_graph(sess, batch, settings):

        feeds = Feeds.Attack(batch)

        attack = Constructor(sess, batch, feeds)

        attack.add_hard_constraint(
            Constraints.L2,
            r_constant=settings["rescale"],
            update_method=settings["constraint_update"],
        )

        attack.add_graph(
            Graphs.SimpleAttack
        )

        attack.add_victim(
            Victim.Model,
            tokens=settings["tokens"],
            beam_width=settings["beam_width"]
        )

        attack.add_loss(Losses.CTCLossV2)
        attack.create_loss_fn()

        attack.add_optimiser(
            Optimisers.AdamOptimiser,
            learning_rate=settings["learning_rate"]
        )

        attack.add_procedure(
            Procedures.UpdateOnDecoding,
            steps=settings["nsteps"],
            decode_step=settings["decode_step"]
        )

        attack.add_outputs(
            Outputs.Base,
            settings["outdir"],
        )

        attack.create_feeds()

        return attack

    outdir = os.path.join(OUTDIR, "ctc2/")

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

    batch_gen = get_standard_batch_generator(settings)

    execute(settings, create_attack_graph, batch_gen)

    log("Finished run.")  # {}.".format(run))


def f6_ctc_beam_search_decoder_run(master_settings):
    """
    Use CTC Loss to optimise some target logits for us. This is quick and simple
    but the deocder confidence scores are usually a lot lower than the example's
    original transcription score.

    :return: None
    """
    def create_attack_graph(sess, batch, settings):
        feeds = Feeds.Attack(batch)

        attack = Constructor(sess, batch, feeds)

        attack.add_hard_constraint(
            Constraints.L2,
            r_constant=settings["rescale"],
            update_method=settings["constraint_update"],
        )

        attack.add_graph(
            Graphs.SimpleAttack
        )

        attack.add_victim(
            Victim.Model,
            tokens=settings["tokens"],
            decoder=settings["decoder_type"],
            beam_width=settings["beam_width"]
        )

        alignment = Constructor(attack.sess, batch, feeds)
        alignment.add_graph(custom_defs.CTCSearchGraph, attack)
        alignment.add_loss(custom_defs.AlignmentLoss)
        alignment.create_loss_fn()
        alignment.add_optimiser(custom_defs.CTCAlignmentOptimiser)

        attack.add_loss(
            Losses.CWMaxDiff,
            alignment.graph.raw_alignments,
            k=settings["kappa"]
        )
        attack.create_loss_fn()
        attack.add_optimiser(
            Optimisers.AdamOptimiser,
            learning_rate=settings["learning_rate"]
        )
        attack.add_procedure(
            custom_defs.CTCAlignmentsUpdateOnDecode,
            alignment_graph=alignment,
            steps=settings["nsteps"],
            decode_step=settings["decode_step"]
        )
        attack.add_outputs(
            Outputs.Base,
            settings["outdir"],
        )

        attack.create_feeds()

        return attack

    # for run in range(0, N_RUNS * 2 + 1, 2):

    outdir = os.path.join(OUTDIR, "maxdiff/")
    outdir = os.path.join(outdir, "ctcalign/")
    outdir = os.path.join(outdir, "kappa_{}/".format(KAPPA))

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
        "kappa": float(KAPPA),
        "decoder_type": "batch",
        "max_examples": MAX_EXAMPLES,
        "max_targets": MAX_TARGETS,
        "max_audio_length": MAX_AUDIO_LENGTH,
    }

    settings.update(master_settings)

    batch_gen = get_standard_batch_generator(settings)

    execute(settings, create_attack_graph, batch_gen)

    log("Finished run.") # {}.".format(run))


def f6_ctc_greedy_search_decoder_run(master_settings):
    """
    Use CTC Loss to optimise some target logits for us. This is quick and simple
    but the deocder confidence scores are usually a lot lower than the example's
    original transcription score.

    :return: None
    """
    def create_attack_graph(sess, batch, settings):
        feeds = Feeds.Attack(batch)

        attack = Constructor(sess, batch, feeds)

        attack.add_hard_constraint(
            Constraints.L2,
            r_constant=settings["rescale"],
            update_method=settings["constraint_update"],
        )

        attack.add_graph(
            Graphs.SimpleAttack
        )

        attack.add_victim(
            Victim.Model,
            tokens=settings["tokens"],
            decoder=settings["decoder_type"],
            beam_width=settings["beam_width"]
        )

        alignment = Constructor(attack.sess, batch, feeds)
        alignment.add_graph(custom_defs.CTCSearchGraph, attack)
        alignment.add_loss(custom_defs.AlignmentLoss)
        alignment.create_loss_fn()
        alignment.add_optimiser(custom_defs.CTCAlignmentOptimiser)

        attack.add_loss(
            Losses.CWMaxDiff,
            alignment.graph.raw_alignments,
            k=settings["kappa"]
        )
        attack.create_loss_fn()
        attack.add_optimiser(
            Optimisers.AdamOptimiser,
            learning_rate=settings["learning_rate"]
        )
        attack.add_procedure(
            custom_defs.CTCAlignmentsUpdateOnDecode,
            alignment_graph=alignment,
            steps=settings["nsteps"],
            decode_step=settings["decode_step"]
        )

        attack.add_outputs(
            Outputs.Base,
            settings["outdir"],
        )

        attack.create_feeds()

        return attack

    # for run in range(0, N_RUNS * 2 + 1, 2):

    outdir = os.path.join(OUTDIR, "maxdiff/")
    outdir = os.path.join(outdir, "ctcalign/")
    outdir = os.path.join(outdir, "kappa_{}/".format(KAPPA))

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
        "kappa": float(KAPPA),
        "decoder_type": "greedy",
        "max_examples": MAX_EXAMPLES,
        "max_targets": MAX_TARGETS,
        "max_audio_length": MAX_AUDIO_LENGTH,
    }

    settings.update(master_settings)

    batch_gen = get_standard_batch_generator(settings)

    execute(
        settings,
        create_attack_graph,
        batch_gen,
    )

    log("Finished run.") # {}.".format(run))


if __name__ == '__main__':

    log("", wrap=True)

    experiments = {
        "ctc": baseline_ctc_run,
        "ctc_v2": baseline_ctc_v2_run,
        "ctcalign_maxdiff_beam": f6_ctc_beam_search_decoder_run,
        "ctcalign_maxdiff_greedy": f6_ctc_greedy_search_decoder_run,
    }

    args(experiments)



