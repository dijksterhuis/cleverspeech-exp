#!/usr/bin/env python3
import os

# attack def imports
from cleverspeech.graph.GraphConstructor import Constructor
from cleverspeech.graph import Constraints
from cleverspeech.graph import VariableGraphs
from cleverspeech.graph import Losses
from cleverspeech.graph import Optimisers
# from cleverspeech.graph import Procedures

from cleverspeech.data.ingress.etl import batch_generators
from cleverspeech.data.ingress import Feeds
from cleverspeech.data.egress.Databases import SingleJsonDB
from cleverspeech.data.egress.Transforms import Standard
from cleverspeech.data.egress.Writers import SingleFileWriter
from cleverspeech.data.egress import Reporting

from cleverspeech.utils.runtime.AttackSpawner import AttackSpawner
from cleverspeech.utils.runtime.ExperimentArguments import args
from cleverspeech.utils.Utils import log, lcomp

from experiments.Perceptual.Synthesis.Synthesisers import Spectral, \
    DeterministicPlusNoise, Additive

from SecEval import VictimAPI as DeepSpeech
import custom_defs


GPU_DEVICE = 0
MAX_PROCESSES = 1
SPAWN_DELAY = 30

AUDIOS_INDIR = "./samples/all/"
TARGETS_PATH = "./samples/cv-valid-test.csv"
OUTDIR = "./adv/synthesis/"
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
        VariableGraphs.Synthesis,
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
        Optimisers.AdamBatchwiseOptimiser,
        learning_rate=settings["learning_rate"]
    )

    attack.add_procedure(
        custom_defs.UpdateOnDecodingSynth,
        steps=settings["nsteps"],
        decode_step=settings["decode_step"]
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
        batch_gen = batch_generators.standard(settings)

        execute(settings, create_attack_graph, batch_gen)

        log("Finished run {}.".format(run))


def freq_harmonic_run(master_settings):

    synth = "freqharmonic"

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
        batch_gen = batch_generators.standard(settings)

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
        batch_gen = batch_generators.standard(settings)

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
        batch_gen = batch_generators.standard(settings)

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
        batch_gen = batch_generators.standard(settings)

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
        batch_gen = batch_generators.standard(settings)

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
        batch_gen = batch_generators.standard(settings)

        execute(settings, create_attack_graph, batch_gen)

        log("Finished run {}.".format(run))


if __name__ == '__main__':

    log("", wrap=True)

    experiments = {
        "stft": spectral_run,
        "additive-inharmonic": inharmonic_run,
        "additive-freq_harmonic": freq_harmonic_run,
        "additive-full_harmonic": full_harmonic_run,
        "detnoise-inharmonic": detnoise_inharmonic_run,
        "detnoise-freq_harmonic": detnoise_freq_harmonic_run,
        "detnoise-full_harmonic": detnoise_full_harmonic_run,
    }

    args(experiments)



