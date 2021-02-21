#!/usr/bin/env python3
import os

# attack def imports
from cleverspeech.graph.GraphConstructor import Constructor
from cleverspeech.graph import Constraints
from cleverspeech.graph.Losses import CTCLoss
from cleverspeech.graph import Optimisers
# from cleverspeech.graph import Procedures
from cleverspeech.graph import Outputs
from cleverspeech.data import Feeds

from cleverspeech.data.etl.batch_generators import get_standard_batch_generator
from cleverspeech.utils.Utils import log, l_map, lcomp, args

from SecEval import VictimAPI as DeepSpeech

from experiments.Perceptual.SynthesisAttacks.Synthesisers import Spectral, \
    DeterministicPlusNoise, Additive

# boilerplate imports

from boilerplate import execute

import custom_defs

GPU_DEVICE = 0
MAX_PROCESSES = 4
SPAWN_DELAY = 30

TOKENS = " abcdefghijklmnopqrstuvwxyz'-"
BEAM_WIDTH = 500

AUDIOS_INDIR = "./samples/all/"
TARGETS_PATH = "./samples/cv-valid-test.csv"
OUTDIR = "./adv/synthesis/"
MAX_EXAMPLES = 100
MAX_TARGETS = 500
MAX_AUDIO_LENGTH = 120000

RESCALE = 0.95
CONSTRAINT_UPDATE = "geom"
LEARNING_RATE = 10
NUMB_STEPS = 10000
DECODING_STEP = 10
BATCH_SIZE = 10
N_OSC = 64

N_RUNS = 5

SYNTHS = {
    "inharmonic": Additive.InHarmonic,
    "freqharmonic": Additive.FreqHarmonic,
    "fullharmonic": Additive.FullyHarmonic,
    "dn_inharmonic": DeterministicPlusNoise.InharmonicPlusPlain,
    "dn_freqharmonic": DeterministicPlusNoise.FreqHarmonicPlusPlain,
    "dn_fullharmonic": DeterministicPlusNoise.FullyHarmonicPlusPlain,
    "stft": Spectral.STFT,
}


def create_attack_graph(sess, batch, settings):

    synth_cls = SYNTHS[settings["synth_cls"]]
    synth = synth_cls(batch, **settings["synth"])

    feeds = Feeds.Attack(batch)
    attack = Constructor(sess, batch, feeds)

    attack.add_hard_constraint(
        Constraints.L2,
        r_constant=settings["rescale"],
        update_method=settings["constraint_update"],
    )

    attack.add_graph(
        custom_defs.SynthesisAttack,
        synth
    )

    attack.add_victim(
        DeepSpeech.Model,
        tokens=settings["tokens"],
        beam_width=settings["beam_width"]
    )

    attack.add_loss(CTCLoss)
    attack.create_loss_fn()

    attack.add_optimiser(
        Optimisers.AdamOptimiser,
        learning_rate=settings["learning_rate"]
    )

    attack.add_procedure(
        custom_defs.UpdateOnDecodingSynth,
        steps=settings["nsteps"],
        decode_step=settings["decode_step"]
    )

    attack.add_outputs(
        Outputs.Base,
        settings["outdir"],
    )

    attack.create_feeds()

    return attack


def inharmonic_run(master_settings):
    synth_cls = "inharmonic"

    for run in range(N_OSC, 1, -4):

        outdir = os.path.join(OUTDIR, synth_cls + "/")
        outdir = os.path.join(outdir, "osc_{}/".format(run))

        settings = {
            "synth_cls": synth_cls,
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
            "synth": {
                "n_osc": N_OSC,
                "initial_hz": 1e-8,
                "frame_length": 512,
                "frame_step": 512,
                "normalise": False,
            },
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

        log("Finished run {}.".format(run))


def freq_harmonic_run(master_settings):

    synth_cls = "freqharmonic"

    for run in range(N_OSC, 1, -4):

        outdir = os.path.join(OUTDIR, synth_cls + "/")
        outdir = os.path.join(outdir, "osc_{}/".format(run))

        settings = {
            "synth_cls": synth_cls,
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
            "synth": {
                "n_osc": N_OSC,
                "initial_hz": 1e-8,
                "frame_length": 512,
                "frame_step": 512,
                "normalise": False,
            },
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

        log("Finished run {}.".format(run))


def full_harmonic_run(master_settings):

    synth_cls = "fullharmonic"

    for run in range(N_OSC, 1, -4):

        outdir = os.path.join(OUTDIR, synth_cls + "/")
        outdir = os.path.join(outdir, "osc_{}/".format(run))

        settings = {
            "synth_cls": synth_cls,
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
            "synth": {
                "n_osc": N_OSC,
                "initial_hz": 1e-8,
                "frame_length": 512,
                "frame_step": 512,
                "normalise": False,
            },
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

        log("Finished run {}.".format(run))


def detnoise_inharmonic_run(master_settings):

    synth_cls = "dn_inharmonic"

    for run in range(N_OSC, 1, -4):

        outdir = os.path.join(OUTDIR, synth_cls + "/")
        outdir = os.path.join(outdir, "osc_{}/".format(run))

        settings = {
            "synth_cls": synth_cls,
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
            "synth": {
                "n_osc": N_OSC,
                "initial_hz": 1e-8,
                "frame_length": 512,
                "frame_step": 512,
                "normalise": False,
            },
            "gpu_device": GPU_DEVICE,
            "max_spawns": MAX_PROCESSES,
            "spawn_delay": SPAWN_DELAY,
            "max_examples": MAX_EXAMPLES,
            "max_targets": MAX_TARGETS,
            "max_audio_length": MAX_AUDIO_LENGTH,
        }

        settings.update(master_settings)
        batch_gen = get_standard_batch_generator(settings)

        execute(settings, create_attack_graph, synth_fn, batch_gen)

        log("Finished run {}.".format(run))


def detnoise_freq_harmonic_run(master_settings):

    synth_cls = "dn_freqharmonic"

    for run in range(N_OSC, 1, -4):

        outdir = os.path.join(OUTDIR, synth_cls + "/")
        outdir = os.path.join(outdir, "osc_{}/".format(run))

        settings = {
            "synth_cls": synth_cls,
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
            "synth": {
                "n_osc": N_OSC,
                "initial_hz": 1e-8,
                "frame_length": 512,
                "frame_step": 512,
                "normalise": False,
            },
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

        log("Finished run {}.".format(run))


def detnoise_full_harmonic_run(master_settings):

    synth_cls = "dn_fullharmonic"

    for run in range(N_OSC, 1, -4):

        outdir = os.path.join(OUTDIR, synth_cls + "/")
        outdir = os.path.join(outdir, "osc_{}/".format(run))

        settings = {
            "synth_cls": synth_cls,
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
            "synth": {
                "n_osc": N_OSC,
                "initial_hz": 1e-8,
                "frame_length": 512,
                "frame_step": 512,
                "normalise": False,
            },
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

        log("Finished run {}.".format(run))


def spectral_run(master_settings):
    """
    CTC Loss attack modified from the original Carlini & Wagner work.

    Using a hard constraint is better for security evaluations, so we ignore the
    L2 distance regularisation term in the optimisation goal.

    TODO: I could probably remove `Base.add_loss()` method...?

    :return: None
    """

    synth_cls = "stft"
    outdir = os.path.join(OUTDIR, "spectral/")

    settings = {
        "synth_cls": synth_cls,
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
        "synth": {
            "frame_step": 256,
            "frame_length": 512,
            "fft_length": 512,
        },
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


def spectral_regularised_run(master_settings):
    """
    CTC Loss attack modified from the original Carlini & Wagner work.

    Using a hard constraint is better for security evaluations, so we ignore the
    L2 distance regularisation term in the optimisation goal.

    TODO: I could probably remove `Base.add_loss()` method...?

    :return: None
    """
    def create_attack_graph(sess, batch, settings):

        synth_cls = SYNTHS[settings["synth_cls"]]
        synth = synth_cls(batch, **settings["synth"])

        feeds = Feeds.Attack(batch)
        attack = Constructor(sess, batch, feeds)

        attack.add_hard_constraint(
            Constraints.L2,
            r_constant=settings["rescale"],
            update_method=settings["constraint_update"],
        )

        attack.add_graph(
            custom_defs.SynthesisAttack,
            synth
        )

        attack.add_victim(
            DeepSpeech.Model,
            tokens=settings["tokens"],
            beam_width=settings["beam_width"]
        )

        attack.add_loss(CTCLoss)
        attack.add_loss(custom_defs.SpectralLoss)
        attack.create_loss_fn()

        attack.add_optimiser(
            Optimisers.AdamOptimiser,
            learning_rate=settings["learning_rate"]
        )

        attack.add_procedure(
            custom_defs.UpdateOnDecodingSynth,
            steps=settings["nsteps"],
            decode_step=settings["decode_step"]
        )

        attack.add_outputs(
            Outputs.Base,
            settings["outdir"],
        )

        attack.create_feeds()

        return attack

    synth_cls = "stft"
    outdir = os.path.join(OUTDIR, "spectral/")

    settings = {
        "synth_cls": synth_cls,
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
        "synth": {
            "frame_step": 256,
            "frame_length": 512,
            "fft_length": 512,
        },
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


if __name__ == '__main__':

    log("", wrap=True)

    experiments = {
        "stft": spectral_run,
        "stft-reg": spectral_regularised_run,
        "inharmonic": inharmonic_run,
        "freq_harmonic": freq_harmonic_run,
        "full_harmonic": full_harmonic_run,
        "detnoise_inharmonic": detnoise_inharmonic_run,
        "detnoise_freq_harmonic": detnoise_freq_harmonic_run,
        "detnoise_full_harmonic": detnoise_full_harmonic_run,
    }

    args(experiments)



