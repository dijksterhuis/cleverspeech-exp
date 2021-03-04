#!/usr/bin/env python3
import os

# attack def imports
from cleverspeech.graph.GraphConstructor import Constructor
from cleverspeech.graph import Constraints
# from cleverspeech.graph import Graphs
from cleverspeech.graph import Losses
from cleverspeech.graph import Optimisers
# from cleverspeech.graph import Procedures
from cleverspeech.graph import Outputs
from cleverspeech.data import Feeds

from cleverspeech.data.etl.batch_generators import get_standard_batch_generator
from cleverspeech.data.Results import SingleJsonDB, SingleFileWriter
from cleverspeech.eval import PerceptualStatsBatch
from cleverspeech.utils.RuntimeUtils import AttackSpawner
from cleverspeech.utils.Utils import log, args

from experiments.Perceptual.SynthesisAttacks.Synthesisers import Spectral, \
    DeterministicPlusNoise, Additive

from SecEval import VictimAPI as DeepSpeech
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

ADDITIVE_N_OSC = 16
ADDITIVE_FRAME_LENGTH = 512
ADDITIVE_FRAME_STEP = 512
ADDITIVE_INITIAL_HZ = 1e-8

SPECTRAL_FRAME_STEP = 256
SPECTRAL_FRAME_LENGTH = 256
SPECTRAL_FFT_LENGTH = 256
SPECTRAL_CONSTANT = 64


# Synthesis Attacks
# ==============================================================================
# Main Question: What happens if we constrain how an adversary can generate
# perturbations instead of only constraining by *how much* perturbation it can
# generate?

SYNTHS = {
    "inharmonic": Additive.InHarmonic,
    "freqharmonic": Additive.FreqHarmonic,
    "fullharmonic": Additive.FullyHarmonic,
    "dn_inharmonic": DeterministicPlusNoise.InharmonicPlusPlain,
    "dn_freqharmonic": DeterministicPlusNoise.FreqHarmonicPlusPlain,
    "dn_fullharmonic": DeterministicPlusNoise.FullyHarmonicPlusPlain,
    "stft": Spectral.STFT,
}


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

    attack.add_loss(Losses.CTCLoss)
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

    synth = "inharmonic"

    for run in range(ADDITIVE_N_OSC, 1, -4):

        outdir = os.path.join(OUTDIR, synth + "/")
        outdir = os.path.join(outdir, "osc_{}/".format(run))

        settings = {
            "synth_cls": synth,
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
                "n_osc": ADDITIVE_N_OSC,
                "initial_hz": ADDITIVE_INITIAL_HZ,
                "frame_length": ADDITIVE_FRAME_LENGTH,
                "frame_step": ADDITIVE_FRAME_STEP,
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

    synth = "freqinharmonic"

    for run in range(ADDITIVE_N_OSC, 1, -4):

        outdir = os.path.join(OUTDIR, synth + "/")
        outdir = os.path.join(outdir, "osc_{}/".format(run))

        settings = {
            "synth_cls": synth,
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
                "n_osc": ADDITIVE_N_OSC,
                "initial_hz": ADDITIVE_INITIAL_HZ,
                "frame_length": ADDITIVE_FRAME_LENGTH,
                "frame_step": ADDITIVE_FRAME_STEP,
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

    synth = "fullharmonic"

    for run in range(ADDITIVE_N_OSC, 1, -4):

        outdir = os.path.join(OUTDIR, synth + "/")
        outdir = os.path.join(outdir, "osc_{}/".format(run))

        settings = {
            "synth_cls": synth,
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
                "n_osc": ADDITIVE_N_OSC,
                "initial_hz": ADDITIVE_INITIAL_HZ,
                "frame_length": ADDITIVE_FRAME_LENGTH,
                "frame_step": ADDITIVE_FRAME_STEP,
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

    synth = "dn_inharmonic"

    for run in range(ADDITIVE_N_OSC, 1, -4):

        outdir = os.path.join(OUTDIR, synth + "/")
        outdir = os.path.join(outdir, "osc_{}/".format(run))

        settings = {
            "synth_cls": synth,
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
                "n_osc": ADDITIVE_N_OSC,
                "initial_hz": ADDITIVE_INITIAL_HZ,
                "frame_length": ADDITIVE_FRAME_LENGTH,
                "frame_step": ADDITIVE_FRAME_STEP,
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


def detnoise_freq_harmonic_run(master_settings):

    synth = "dn_freqharmonic"

    for run in range(ADDITIVE_N_OSC, 1, -4):
        outdir = os.path.join(OUTDIR, synth + "/")
        outdir = os.path.join(outdir, "osc_{}/".format(run))

        settings = {
            "synth_cls": synth,
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
                "n_osc": ADDITIVE_N_OSC,
                "initial_hz": ADDITIVE_INITIAL_HZ,
                "frame_length": ADDITIVE_FRAME_LENGTH,
                "frame_step": ADDITIVE_FRAME_STEP,
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

    synth = "dn_fullharmonic"

    for run in range(ADDITIVE_N_OSC, 1, -4):

        outdir = os.path.join(OUTDIR, synth + "/")
        outdir = os.path.join(outdir, "osc_{}/".format(run))

        settings = {
            "synth_cls": synth,
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
                "n_osc": ADDITIVE_N_OSC,
                "initial_hz": ADDITIVE_INITIAL_HZ,
                "frame_length": ADDITIVE_FRAME_LENGTH,
                "frame_step": ADDITIVE_FRAME_STEP,
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

    def run_generator(x, n):
        for i in range(1, n+1):
            if i == 0:
                yield x
            else:
                yield x
                x *= 2

    runs = lcomp(run_generator(SPECTRAL_CONSTANT, 8))

    synth = "stft"

    for run in runs:

        outdir = os.path.join(OUTDIR, synth + "/")
        outdir = os.path.join(outdir, "run_{}/".format(run))

        settings = {
            "synth_cls": synth,
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
            "learning_rate": 100,
            "synth": {
                "frame_step": run,
                "frame_length": run,
                "fft_length": run * 2,
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


if __name__ == '__main__':

    log("", wrap=True)

    experiments = {
        "stft": spectral_run,
        "inharmonic": inharmonic_run,
        "freq_harmonic": freq_harmonic_run,
        "full_harmonic": full_harmonic_run,
        "detnoise_inharmonic": detnoise_inharmonic_run,
        "detnoise_freq_harmonic": detnoise_freq_harmonic_run,
        "detnoise_full_harmonic": detnoise_full_harmonic_run,
    }

    args(experiments)



