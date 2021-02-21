#!/usr/bin/env python3
import os
import numpy as np

from cleverspeech.graph.GraphConstructor import Constructor
from cleverspeech.graph import Constraints
from cleverspeech.graph import Graphs
from cleverspeech.graph import Losses
from cleverspeech.graph import Optimisers
from cleverspeech.graph import Procedures
from cleverspeech.graph import Outputs
from cleverspeech.data import Feeds

from cleverspeech.data.etl.batch_generators import get_standard_batch_generator
from cleverspeech.data.etl.batch_generators import get_dense_batch_factory
from cleverspeech.utils.Utils import log, args

from SecEval import VictimAPI as DeepSpeech

from boilerplate import execute

# local attack classes
import custom_defs

GPU_DEVICE = 0
MAX_PROCESSES = 3
SPAWN_DELAY = 30

TOKENS = " abcdefghijklmnopqrstuvwxyz'-"
BEAM_WIDTH = 500

AUDIOS_INDIR = "./samples/all/"
TARGETS_PATH = "./samples/cv-valid-test.csv"
OUTDIR = "./adv/ctc-alignments/"

# targets search parameters
MAX_EXAMPLES = 1000
MAX_TARGETS = 1000
MAX_AUDIO_LENGTH = 120000

RESCALE = 0.95
CONSTRAINT_UPDATE = "geom"
LEARNING_RATE = 10
NUMB_STEPS = 20000
DECODING_STEP = 10
BATCH_SIZE = 10

# extreme run settings
LOSS_UPDATE_THRESHOLD = 10.0
LOSS_UPDATE_NUMB_STEPS = 50000

N_RUNS = 1

KAPPA_HYPERS = [[k/(10 ** l) for k in [np.exp2(x) for x in range(0, 4)]] for l in range(1, 3)]
KAPPA = 0.02


# Max Diff Adaptive Kappa Confidence Attacks
# ==============================================================================
# Main ideas:
# Set kappa kappa as a constant per frame instead of a single constant over all
# frames.
#
# Additional work to use CTC Loss (and rCTC variant) as a regulariser to ensure
# target alignments are valid CTC alignments.


def adaptive_kappa_ctc_dense_run(master_settings):
    """
    Currently broken.
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

        attack.add_loss(
            custom_defs.AdaptiveKappaCWMaxDiff,
            attack.graph.placeholders.targets,
            k=settings["kappa"]
        )
        attack.add_loss(
            Losses.CTCLoss,
        )
        attack.create_loss_fn()
        attack.add_optimiser(
            Optimisers.AdamOptimiser,
            learning_rate=settings["learning_rate"]
        )
        attack.add_procedure(
            Procedures.UpdateOnDecoding,
            steps=settings["nsteps"],
            decode_step=settings["decode_step"],
            loss_update_idx=0,
        )
        attack.add_outputs(
            Outputs.Base,
            settings["outdir"],
        )
        attack.create_feeds()

        return attack

    outdir = os.path.join(OUTDIR, "dense/")
    outdir = os.path.join(outdir, "adaptivekappa-ctc/")
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
    }

    settings.update(master_settings)
    batch_gen = get_dense_batch_factory(settings)

    execute(settings, create_attack_graph, batch_gen)

    log("Finished run {}.".format(KAPPA))


def adaptive_kappa_rctc_dense_run(master_settings):
    """
    Currently broken.
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

        attack.add_loss(
            custom_defs.AdaptiveKappaCWMaxDiff,
            attack.graph.placeholders.targets,
            k=settings["kappa"]
        )
        attack.add_loss(
            custom_defs.RepeatsCTCLoss,
            alignment=attack.graph.placeholders.targets,
        )
        attack.create_loss_fn()

        attack.add_optimiser(
            Optimisers.AdamOptimiser,
            learning_rate=settings["learning_rate"]
        )
        attack.add_procedure(
            Procedures.UpdateOnDecoding,
            steps=settings["nsteps"],
            decode_step=settings["decode_step"],
            loss_update_idx=0,
        )
        attack.add_outputs(
            Outputs.Base,
            settings["outdir"],
        )

        attack.create_feeds()

        return attack

    outdir = os.path.join(OUTDIR, "dense/")
    outdir = os.path.join(outdir, "adaptivekappa-rctc/")
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
    }

    settings.update(master_settings)
    batch_gen = get_dense_batch_factory(settings)

    execute(settings, create_attack_graph, batch_gen)

    log("Finished run {}.".format(KAPPA))


def adaptive_kappa_dense_run(master_settings):
    """
    MRMaxDiff (extension of CWMaxDiff) with a dense alignment.
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

        attack.add_loss(
            custom_defs.AdaptiveKappaCWMaxDiff,
            attack.graph.placeholders.targets,
            k=settings["kappa"]
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

    outdir = os.path.join(OUTDIR, "dense/")
    outdir = os.path.join(outdir, "adaptivekappa/")
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
    batch_gen = get_dense_batch_factory(settings)

    execute(settings, create_attack_graph, batch_gen)

    log("Finished run {}.".format(KAPPA))


def adaptive_kappa_ctc_sparse_run(master_settings):
    """
    MRMaxDiff (extension of CWMaxDiff) with CTCLoss and sparse alignment.

    Aim: Optimise MRMaxDiff whilst regularising with CTCLoss. It seems that this
    actually does the opposite, using MRMaxDiff as a regulariser for CTCLoss to
    force one the specific alignment.

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
            DeepSpeech.Model,
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
            custom_defs.AdaptiveKappaCWMaxDiff,
            alignment.graph.target_alignments,
            k=settings["kappa"],
        )
        attack.add_loss(
            Losses.CTCLoss,
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
            decode_step=settings["decode_step"],
            loss_update_idx=0,
        )
        attack.add_outputs(
            Outputs.Base,
            settings["outdir"],
        )
        attack.create_feeds()

        return attack

    outdir = os.path.join(OUTDIR, "sparse/")
    outdir = os.path.join(outdir, "adaptivekappa-ctc/")
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

    log("Finished run {}.".format(KAPPA))


def adaptive_kappa_rctc_sparse_run(master_settings):
    """
    MRMaxDiff with CTCLoss for a single alignment using a sparse alignment as
    target.

    Aim: Optimise MRMaxDiff whilst regularising with CTCLoss. It seems that this
    actually does the opposite, using MRMaxDiff as a regulariser for CTCLoss to
    force one the specific alignment.

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
            DeepSpeech.Model,
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
            custom_defs.AdaptiveKappaCWMaxDiff,
            alignment.graph.target_alignments,
            k=settings["kappa"],
        )
        attack.add_loss(
            custom_defs.RepeatsCTCLoss,
            alignment=alignment.graph.target_alignments,
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
            decode_step=settings["decode_step"],
            loss_update_idx=0,
        )
        attack.add_outputs(
            Outputs.Base,
            settings["outdir"],
        )
        attack.create_feeds()

        return attack

    outdir = os.path.join(OUTDIR, "sparse/")
    outdir = os.path.join(outdir, "adaptivekappa-rctc/")
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

    log("Finished run {}.".format(KAPPA))


def adaptive_kappa_sparse_run(master_settings):
    """
    MRMaxDiff (extension of CWMaxDiff) with with a sparse alignment.
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
            DeepSpeech.Model,
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
            custom_defs.AdaptiveKappaCWMaxDiff,
            alignment.graph.target_alignments,
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

    outdir = os.path.join(OUTDIR, "sparse/")
    outdir = os.path.join(outdir, "adaptivekappa/")
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
    }

    settings.update(master_settings)
    batch_gen = get_standard_batch_generator(settings)

    execute(settings, create_attack_graph, batch_gen)

    log("Finished run {}.".format(KAPPA))


if __name__ == '__main__':

    experiments = {
        "sparse-adaptivekappa-ctc": adaptive_kappa_ctc_sparse_run,
        "dense-adaptivekappa-ctc": adaptive_kappa_ctc_dense_run,
        "sparse-adaptivekappa-rctc": adaptive_kappa_rctc_sparse_run,
        "dense-adaptivekappa-rctc": adaptive_kappa_rctc_dense_run,
        "sparse-adaptivekappa": adaptive_kappa_sparse_run,
        "dense-adaptivekappa": adaptive_kappa_dense_run,
    }

    args(experiments)

