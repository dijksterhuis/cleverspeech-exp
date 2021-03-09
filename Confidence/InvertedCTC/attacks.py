#!/usr/bin/env python3
import os

from cleverspeech.graph.GraphConstructor import Constructor
from cleverspeech.graph import Constraints
from cleverspeech.graph import Graphs
from cleverspeech.graph import Losses
from cleverspeech.graph import Optimisers
from cleverspeech.graph import Procedures
from cleverspeech.graph import Outputs
from cleverspeech.graph.CTCAlignmentSearch import create_tf_ctc_alignment_search_graph

from cleverspeech.data import Feeds
from cleverspeech.data.etl.batch_generators import get_standard_batch_generator
from cleverspeech.data.etl.batch_generators import get_dense_batch_factory
from cleverspeech.data.etl.batch_generators import get_sparse_batch_generator
from cleverspeech.data.Results import SingleJsonDB, SingleFileWriter
from cleverspeech.eval import PerceptualStatsBatch
from cleverspeech.utils.RuntimeUtils import AttackSpawner
from cleverspeech.utils.Utils import log, args

# Victim model import
from SecEval import VictimAPI as DeepSpeech


GPU_DEVICE = 0
MAX_PROCESSES = 3
SPAWN_DELAY = 30

AUDIOS_INDIR = "./samples/all/"
TARGETS_PATH = "./samples/cv-valid-test.csv"
OUTDIR = "./adv/antictc/"
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


def create_adaptive_kappa_attack_graph(sess, batch, settings):

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
        DeepSpeech.Model,
        tokens=settings["tokens"],
        decoder=settings["decoder_type"],
        beam_width=settings["beam_width"]
    )

    attack.add_loss(
        Losses.AntiCTC,
        alignment=attack.graph.placeholders.targets,
        weight_settings=(1 / 100, 1 / 100)
    )

    attack.add_loss(
        Losses.AdaptiveKappaMaxDiff,
        attack.graph.placeholders.targets,
        k=1.0,
    )
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


def dense_adaptive_kappa_run(master_settings):
    """
    """

    outdir = os.path.join(OUTDIR, "adaptive_kappa/")
    outdir = os.path.join(outdir, "dense/")

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
        "decoder_type": "batch",
        "max_examples": MAX_EXAMPLES,
        "max_targets": MAX_TARGETS,
        "max_audio_length": MAX_AUDIO_LENGTH,
    }

    settings.update(master_settings)
    batch_gen = get_dense_batch_factory(settings)
    execute(settings, create_adaptive_kappa_attack_graph, batch_gen)
    log("Finished run.")


def sparse_adaptive_kappa_run(master_settings):
    """
    """
    outdir = os.path.join(OUTDIR, "adaptive_kappa/")
    outdir = os.path.join(outdir, "sparse/")

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
        "decoder_type": "batch",
        "max_examples": MAX_EXAMPLES,
        "max_targets": MAX_TARGETS,
        "max_audio_length": MAX_AUDIO_LENGTH,
    }

    settings.update(master_settings)
    batch_gen = get_sparse_batch_generator(settings)
    execute(settings, create_adaptive_kappa_attack_graph, batch_gen)
    log("Finished run.")


def ctcalign_adaptive_kappa_run(master_settings):
    """
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
            DeepSpeech.Model,
            tokens=settings["tokens"],
            decoder=settings["decoder_type"],
            beam_width=settings["beam_width"]
        )

        alignment = create_tf_ctc_alignment_search_graph(attack, batch, feeds)

        attack.add_loss(
            Losses.AntiCTC,
            alignment=alignment.graph.target_alignments,
            weight_settings=(1/100, 1/100)
        )

        attack.add_loss(
            Losses.AdaptiveKappaMaxDiff,
            alignment.graph.target_alignments,
            k=1.0,
        )

        attack.create_loss_fn()
        attack.add_optimiser(
            Optimisers.AdamOptimiser,
            learning_rate=settings["learning_rate"]
        )
        attack.add_procedure(
            Procedures.CTCAlignUpdateOnDecode,
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

    outdir = os.path.join(OUTDIR, "adaptive_kappa/")
    outdir = os.path.join(outdir, "ctcalign/")

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
        "decoder_type": "batch",
        "max_examples": MAX_EXAMPLES,
        "max_targets": MAX_TARGETS,
        "max_audio_length": MAX_AUDIO_LENGTH,
    }

    settings.update(master_settings)
    batch_gen = get_standard_batch_generator(settings)
    execute(settings, create_attack_graph, batch_gen)
    log("Finished run.")


if __name__ == '__main__':

    experiments = {
        "dense-adaptive-kappa": dense_adaptive_kappa_run,
        "sparse-adaptive-kappa": sparse_adaptive_kappa_run,
        "ctcalign-adaptive-kappa": ctcalign_adaptive_kappa_run,
    }

    args(experiments)

