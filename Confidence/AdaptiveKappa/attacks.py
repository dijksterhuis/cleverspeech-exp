#!/usr/bin/env python3
import os
import numpy as np

from cleverspeech.graph.GraphConstructor import Constructor
from cleverspeech.graph import Constraints
from cleverspeech.graph import Graphs
from cleverspeech.graph import Losses
from cleverspeech.graph import Optimisers
from cleverspeech.graph import Procedures
from cleverspeech.graph.CTCAlignmentSearch import create_tf_ctc_alignment_search_graph

from cleverspeech.data.ingress.etl import batch_generators
from cleverspeech.data.ingress import Feeds
from cleverspeech.data.egress.Databases import SingleJsonDB
from cleverspeech.data.egress.Transforms import Standard
from cleverspeech.data.egress.Writers import SingleFileWriter
from cleverspeech.data.egress.eval import PerceptualStatsBatch

from cleverspeech.utils.RuntimeUtils import AttackSpawner
from cleverspeech.utils.Utils import log, args

# victim model import
from SecEval import VictimAPI as DeepSpeech


GPU_DEVICE = 0
MAX_PROCESSES = 1
SPAWN_DELAY = 30

AUDIOS_INDIR = "./samples/all/"
TARGETS_PATH = "./samples/cv-valid-test.csv"
OUTDIR = "./adv/confidence/adaptivekappa/"
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

KAPPA_HYPERS = [
    [k/(10 ** l) for k in [np.exp2(x) for x in range(0, 4)]] for l in range(1, 3)
]
KAPPA = 0.5


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
    PerceptualStatsBatch.batch_generate_statistic_file(settings["outdir"])


def create_standard_attack_graph(sess, batch, settings):

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
        Losses.AdaptiveKappaMaxDiff,
        attack.graph.placeholders.targets,
        k=settings["kappa"]
    )

    if settings["additional_loss"] == "ctc":
        attack.add_loss(
            Losses.CTCLoss,
        )

    elif settings["additional_loss"] == "rctc":
        attack.add_loss(
            Losses.RepeatsCTCLoss,
            alignment=attack.graph.placeholders.targets,
        )

    elif settings["additional_loss"] is "none":
        pass

    else:
        raise NotImplementedError

    attack.create_loss_fn()
    attack.add_optimiser(
        Optimisers.AdamOptimiser,
        learning_rate=settings["learning_rate"]
    )
    attack.add_procedure(
        Procedures.UpdateOnDecoding,
        steps=settings["nsteps"],
        decode_step=settings["decode_step"],
        loss_update_idx=[0],
    )
    attack.create_feeds()

    return attack


def create_ctcalign_attack_graph(sess, batch, settings):

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
        Losses.AdaptiveKappaMaxDiff,
        alignment.graph.target_alignments,
        k=settings["kappa"]
    )

    if settings["additional_loss"] == "ctc":
        attack.add_loss(
            Losses.CTCLoss,
        )

    elif settings["additional_loss"] == "rctc":
        attack.add_loss(
            Losses.RepeatsCTCLoss,
            alignment=alignment.graph.target_alignments,
        )

    elif settings["additional_loss"] is "none":
        pass

    else:
        raise NotImplementedError

    attack.create_loss_fn()
    attack.add_optimiser(
        Optimisers.AdamOptimiser,
        learning_rate=settings["learning_rate"]
    )
    attack.add_procedure(
        Procedures.CTCAlignUpdateOnDecode,
        alignment,
        steps=settings["nsteps"],
        decode_step=settings["decode_step"],
        loss_update_idx=[0],
    )
    attack.create_feeds()

    return attack


def ctc_dense_run(master_settings):
    """
    Currently broken.
    """

    additional_loss = "ctc"

    outdir = os.path.join(OUTDIR, "dense/")
    outdir = os.path.join(outdir, "{}/".format(additional_loss))
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
        "kappa": KAPPA,
        "decoder_type": "batch",
        "max_examples": MAX_EXAMPLES,
        "max_targets": MAX_TARGETS,
        "max_audio_length": MAX_AUDIO_LENGTH,
        "additional_loss": additional_loss,
    }

    settings.update(master_settings)
    batch_gen = batch_generators.dense(settings)
    execute(settings, create_standard_attack_graph, batch_gen)
    log("Finished run {}.".format(KAPPA))


def rctc_dense_run(master_settings):
    """
    Currently broken.
    """

    additional_loss = "rctc"

    outdir = os.path.join(OUTDIR, "dense/")
    outdir = os.path.join(outdir, "{}/".format(additional_loss))
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
        "kappa": KAPPA,
        "decoder_type": "batch",
        "max_examples": MAX_EXAMPLES,
        "max_targets": MAX_TARGETS,
        "max_audio_length": MAX_AUDIO_LENGTH,
        "additional_loss": additional_loss,
    }

    settings.update(master_settings)
    batch_gen = batch_generators.dense(settings)
    execute(settings, create_standard_attack_graph, batch_gen)
    log("Finished run {}.".format(KAPPA))


def dense_run(master_settings):

    additional_loss = "none"

    outdir = os.path.join(OUTDIR, "dense/")
    outdir = os.path.join(outdir, "{}/".format(additional_loss))
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
        "kappa": KAPPA,
        "decoder_type": "batch",
        "max_examples": MAX_EXAMPLES,
        "max_targets": MAX_TARGETS,
        "max_audio_length": MAX_AUDIO_LENGTH,
        "additional_loss": additional_loss,
    }

    settings.update(master_settings)
    batch_gen = batch_generators.dense(settings)
    execute(settings, create_standard_attack_graph, batch_gen)
    log("Finished run {}.".format(KAPPA))


def ctc_sparse_run(master_settings):
    """
    Currently broken.
    """

    additional_loss = "ctc"

    outdir = os.path.join(OUTDIR, "sparse/")
    outdir = os.path.join(outdir, "{}/".format(additional_loss))
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
        "kappa": KAPPA,
        "decoder_type": "batch",
        "max_examples": MAX_EXAMPLES,
        "max_targets": MAX_TARGETS,
        "max_audio_length": MAX_AUDIO_LENGTH,
        "additional_loss": additional_loss,
    }

    settings.update(master_settings)
    batch_gen = batch_generators.sparse(settings)
    execute(settings, create_standard_attack_graph, batch_gen)
    log("Finished run {}.".format(KAPPA))


def rctc_sparse_run(master_settings):
    """
    Currently broken.
    """
    additional_loss = "rctc"

    outdir = os.path.join(OUTDIR, "sparse/")
    outdir = os.path.join(outdir, "{}/".format(additional_loss))
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
        "kappa": KAPPA,
        "decoder_type": "batch",
        "max_examples": MAX_EXAMPLES,
        "max_targets": MAX_TARGETS,
        "max_audio_length": MAX_AUDIO_LENGTH,
        "additional_loss": additional_loss,
    }

    settings.update(master_settings)
    batch_gen = batch_generators.sparse(settings)
    execute(settings, create_standard_attack_graph, batch_gen)
    log("Finished run {}.".format(KAPPA))


def sparse_run(master_settings):
    additional_loss = "none"

    outdir = os.path.join(OUTDIR, "sparse/")
    outdir = os.path.join(outdir, "{}/".format(additional_loss))
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
        "kappa": KAPPA,
        "decoder_type": "batch",
        "max_examples": MAX_EXAMPLES,
        "max_targets": MAX_TARGETS,
        "max_audio_length": MAX_AUDIO_LENGTH,
        "additional_loss": additional_loss,
    }

    settings.update(master_settings)
    batch_gen = batch_generators.sparse(settings)
    execute(settings, create_standard_attack_graph, batch_gen)
    log("Finished run {}.".format(KAPPA))


def ctc_ctcalign_run(master_settings):
    additional_loss = "ctc"

    outdir = os.path.join(OUTDIR, "ctcalign/")
    outdir = os.path.join(outdir, "{}/".format(additional_loss))
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
        "kappa": KAPPA,
        "decoder_type": "batch",
        "max_examples": MAX_EXAMPLES,
        "max_targets": MAX_TARGETS,
        "max_audio_length": MAX_AUDIO_LENGTH,
        "additional_loss": additional_loss,
    }

    settings.update(master_settings)
    batch_gen = batch_generators.standard(settings)
    execute(settings, create_ctcalign_attack_graph, batch_gen)
    log("Finished run {}.".format(KAPPA))


def rctc_ctcalign_run(master_settings):
    additional_loss = "rctc"

    outdir = os.path.join(OUTDIR, "ctcalign/")
    outdir = os.path.join(outdir, "{}/".format(additional_loss))
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
        "kappa": KAPPA,
        "decoder_type": "batch",
        "max_examples": MAX_EXAMPLES,
        "max_targets": MAX_TARGETS,
        "max_audio_length": MAX_AUDIO_LENGTH,
        "additional_loss": additional_loss,
    }

    settings.update(master_settings)
    batch_gen = batch_generators.standard(settings)
    execute(settings, create_ctcalign_attack_graph, batch_gen)
    log("Finished run {}.".format(KAPPA))


def ctcalign_run(master_settings):
    additional_loss = "none"

    outdir = os.path.join(OUTDIR, "ctcalign/")
    outdir = os.path.join(outdir, "{}/".format(additional_loss))
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
        "kappa": KAPPA,
        "decoder_type": "batch",
        "max_examples": MAX_EXAMPLES,
        "max_targets": MAX_TARGETS,
        "max_audio_length": MAX_AUDIO_LENGTH,
        "additional_loss": additional_loss,
    }

    settings.update(master_settings)
    batch_gen = batch_generators.standard(settings)
    execute(settings, create_ctcalign_attack_graph, batch_gen)
    log("Finished run {}.".format(KAPPA))


if __name__ == '__main__':

    experiments = {
        "sparse-rctc": rctc_sparse_run,
        "sparse-ctc": ctc_sparse_run,
        "sparse-none": sparse_run,
        "dense-rctc": rctc_dense_run,
        "dense-ctc": ctc_dense_run,
        "dense-none": dense_run,
        "ctcalign-rctc": rctc_ctcalign_run,
        "ctcalign-ctc": ctc_ctcalign_run,
        "ctcalign-none": ctcalign_run,
    }

    args(experiments)

