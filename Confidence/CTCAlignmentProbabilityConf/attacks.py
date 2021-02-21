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
OUTDIR = "./adv/vibertish/"

# targets search parameters
MAX_EXAMPLES = 100
MAX_TARGETS = 1000
MAX_AUDIO_LENGTH = 120000

RESCALE = 0.95
CONSTRAINT_UPDATE = "geom"
LEARNING_RATE = 10
NUMB_STEPS = 20000
DECODING_STEP = 10
BATCH_SIZE = 10

N_RUNS = 1


# VIBERT-ish
# ==============================================================================
# Main idea: We should optimise a specific alignment to become more likely than
# all others instead of optimising for individual class labels per frame.


def vibertish_fwd_only_dense_run(master_settings):
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

        attack.add_loss(
            custom_defs.FwdOnlyVibertish,
            attack.graph.placeholders.targets,
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

    outdir = os.path.join(OUTDIR, "fwd_only/")
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

    execute(settings, create_attack_graph, batch_gen)

    log("Finished run.")


def vibertish_back_only_dense_run(master_settings):
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

        attack.add_loss(
            custom_defs.BackOnlyVibertish,
            attack.graph.placeholders.targets,
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

    outdir = os.path.join(OUTDIR, "back_only/")
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

    execute(settings, create_attack_graph, batch_gen)

    log("Finished run.")


def vibertish_fwd_plus_back_dense_run(master_settings):
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

        attack.add_loss(
            custom_defs.FwdPlusBackVibertish,
            attack.graph.placeholders.targets,
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

    outdir = os.path.join(OUTDIR, "fwd_plus_back/")
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

    execute(settings, create_attack_graph, batch_gen)

    log("Finished run.")


def vibertish_fwd_mult_back_dense_run(master_settings):
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

        attack.add_loss(
            custom_defs.FwdMultBackVibertish,
            attack.graph.placeholders.targets,
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

    outdir = os.path.join(OUTDIR, "fwd_mult_back/")
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

    execute(settings, create_attack_graph, batch_gen)

    log("Finished run.")


def vibertish_fwd_only_sparse_run(master_settings):
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

        alignment = Constructor(attack.sess, batch, feeds)
        alignment.add_graph(custom_defs.CTCSearchGraph, attack)
        alignment.add_loss(custom_defs.AlignmentLoss)
        alignment.create_loss_fn()
        alignment.add_optimiser(custom_defs.CTCAlignmentOptimiser)

        attack.add_loss(
            custom_defs.FwdOnlyVibertish,
            alignment.graph.target_alignments,
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

    outdir = os.path.join(OUTDIR, "fwd_only/")
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
    batch_gen = get_standard_batch_generator(settings)

    execute(settings, create_attack_graph, batch_gen)

    log("Finished run.")


def vibertish_back_only_sparse_run(master_settings):
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

        alignment = Constructor(attack.sess, batch, feeds)
        alignment.add_graph(custom_defs.CTCSearchGraph, attack)
        alignment.add_loss(custom_defs.AlignmentLoss)
        alignment.create_loss_fn()
        alignment.add_optimiser(custom_defs.CTCAlignmentOptimiser)

        attack.add_loss(
            custom_defs.BackOnlyVibertish,
            alignment.graph.target_alignments,
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

    outdir = os.path.join(OUTDIR, "back_only/")
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
    batch_gen = get_standard_batch_generator(settings)

    execute(settings, create_attack_graph, batch_gen)

    log("Finished run.")


def vibertish_fwd_plus_back_sparse_run(master_settings):
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

        alignment = Constructor(attack.sess, batch, feeds)
        alignment.add_graph(custom_defs.CTCSearchGraph, attack)
        alignment.add_loss(custom_defs.AlignmentLoss)
        alignment.create_loss_fn()
        alignment.add_optimiser(custom_defs.CTCAlignmentOptimiser)

        attack.add_loss(
            custom_defs.FwdPlusBackVibertish,
            alignment.graph.target_alignments,
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

    outdir = os.path.join(OUTDIR, "fwd_plus_back/")
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
    batch_gen = get_standard_batch_generator(settings)

    execute(settings, create_attack_graph, batch_gen)

    log("Finished run.")


def vibertish_fwd_mult_back_sparse_run(master_settings):
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

        alignment = Constructor(attack.sess, batch)
        alignment.add_graph(custom_defs.CTCSearchGraph, attack)
        alignment.add_loss(custom_defs.AlignmentLoss)
        alignment.create_loss_fn()
        alignment.add_optimiser(custom_defs.CTCAlignmentOptimiser)

        attack.add_loss(
            custom_defs.FwdMultBackVibertish,
            alignment.graph.target_alignments,
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

    outdir = os.path.join(OUTDIR, "fwd_mult_back/")
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
    batch_gen = get_standard_batch_generator(settings)

    execute(settings, create_attack_graph, batch_gen)

    log("Finished run.")


if __name__ == '__main__':

    experiments = {
        "dense-vibertish-fwd": vibertish_fwd_only_dense_run,
        "dense-vibertish-back": vibertish_back_only_sparse_run,
        "dense-vibertish-fwdplusback": vibertish_fwd_plus_back_sparse_run,
        "dense-vibertish-fwdmultback": vibertish_fwd_mult_back_sparse_run,
        "sparse-vibertish-fwd": vibertish_fwd_only_sparse_run,
        "sparse-vibertish-back": vibertish_back_only_sparse_run,
        "sparse-vibertish-fwdplusback": vibertish_fwd_plus_back_sparse_run,
        "sparse-vibertish-fwdmultback": vibertish_fwd_mult_back_sparse_run,
    }

    args(experiments)

