#!/usr/bin/env python3
import os
import sys
import itertools
import argparse
#import pynvml

from cleverspeech.Attacks.Base import Constructor
from cleverspeech.Attacks import Constraints
from cleverspeech.Attacks import Graphs
from cleverspeech.Attacks import Losses
from cleverspeech.Attacks import Optimisers
from cleverspeech.Attacks import Procedures
from cleverspeech.Models import DeepSpeech
from AdditiveSynthesis.Synthesis import Additive, DeterministicPlusNoise
from AdditiveSynthesis.Synthesis import Plain
from cleverspeech.Utils import log

from . import Boilerplate
from . import Globals


def attack_graph(sess, batch, synth, settings):

    attack = Constructor(sess, batch)

    attack.add_hard_constraint(Constraints.L2, r_constant=settings["rescale"])
    attack.add_graph(Graphs.SynthesisAttack, synthesis=synth)
    attack.add_victim(
        DeepSpeech.Model,
        tokens=settings["tokens"],
        beam_width=settings["beam_width"]
    )
    attack.add_adversarial_loss(Losses.CTCLoss)
    attack.create_loss_fn()
    attack.add_optimiser(
        Optimisers.GradientDescentOptimiser,
        learning_rate=settings["learning_rate"]
    )
    attack.add_procedure(
        Procedures.UpdateBound,
        steps=settings["nsteps"],
        decode_step=settings["decode_step"]
    )

    return attack


def baseline_run(gpu_device, batch_size):

    for test_idx, (target_phrase) in enumerate(Globals.TARGETS):
        target_idx = Globals.TARGETS.index(target_phrase)

        outdir = os.path.join(Globals.OUTDIR, "ctc_hardl2/")
        outdir = os.path.join(outdir, "baseline/")
        outdir = os.path.join(outdir, "targid_{}/".format(target_idx))

        settings = {
            "indir": Globals.INDIR,
            "outdir": outdir,
            "batch_size": batch_size,
            "target": target_phrase,
            "tokens": Globals.TOKENS,
            "nsteps": Globals.NUMB_STEPS,
            "decode_step": Globals.DECODING_STEP,
            "beam_width": Globals.BEAM_WIDTH,
            "rescale": 0.95,
            "learning_rate": 10.0,
            "exp_epsilon": None,
            "gpu_device": gpu_device,
            "fn": {
                "synth": Plain.Plain,
                "attack": attack_graph,
            },
            "synth": {}

        }

        Boilerplate.run(settings)

        log("Finished attacking for current baseline settings!")
    log("Finished baseline run!")


def inharmonic_run(gpu_device, batch_size):

    eval_gen = itertools.product(Globals.TARGETS, Globals.N_OSC)

    for test_idx, (target_phrase, n_osc) in enumerate(eval_gen):
        target_idx = Globals.TARGETS.index(target_phrase)

        outdir = os.path.join(Globals.OUTDIR, "ctc_hardl2/")
        outdir = os.path.join(outdir, "inharmonic/")
        outdir = os.path.join(outdir, "targid_{}/".format(target_idx))
        outdir = os.path.join(outdir, "osc_{}/".format(n_osc))

        settings = {
            "indir": Globals.INDIR,
            "outdir": outdir,
            "batch_size": batch_size,
            "target": target_phrase,
            "tokens": Globals.TOKENS,
            "nsteps": Globals.NUMB_STEPS,
            "decode_step": Globals.DECODING_STEP,
            "beam_width": Globals.BEAM_WIDTH,
            "rescale": 0.95,
            "learning_rate": 250.0,
            "exp_epsilon": None,
            "gpu_device": gpu_device,
            "fn": {
                "synth": Additive.InHarmonic,
                "attack": attack_graph,
            },
            "synth": {
                "n_osc": n_osc,
                "initial_hz": 1e-8,
                "frame_length": 512,
                "frame_step": 512,
                "normalise": False,
            },

        }

        Boilerplate.run(settings)

        log("Finished attacking for current inharmonic settings!")
    log("Finished inharmonic run!")


def frequency_harmonic_run(gpu_device, batch_size):

    eval_gen = itertools.product(Globals.TARGETS, Globals.N_OSC)

    for test_idx, (target_phrase, n_osc) in enumerate(eval_gen):
        target_idx = Globals.TARGETS.index(target_phrase)

        outdir = os.path.join(Globals.OUTDIR, "ctc_hardl2/")
        outdir = os.path.join(outdir, "freq-harmonic/")
        outdir = os.path.join(outdir, "targid_{}/".format(target_idx))
        outdir = os.path.join(outdir, "osc_{}/".format(n_osc))

        settings = {
            "indir": Globals.INDIR,
            "outdir": outdir,
            "batch_size": batch_size,
            "target": target_phrase,
            "tokens": Globals.TOKENS,
            "nsteps": Globals.NUMB_STEPS,
            "decode_step": Globals.DECODING_STEP,
            "beam_width": Globals.BEAM_WIDTH,
            "rescale": 0.95,
            "learning_rate": 250.0,
            "exp_epsilon": None,
            "gpu_device": gpu_device,
            "fn": {
                "synth": Additive.FreqHarmonic,
                "attack": attack_graph,
            },
            "synth": {
                "n_osc": n_osc,
                "initial_hz": 1e-8,
                "frame_length": 512,
                "frame_step": 512,
                "normalise": False,
            }
        }

        Boilerplate.run(settings)

        log("Finished attacking for current harmonic settings!")
    log("Finished harmonic run!")


def fully_harmonic_run(gpu_device, batch_size):

    eval_gen = itertools.product(Globals.TARGETS, Globals.N_OSC)

    for test_idx, (target_phrase, n_osc) in enumerate(eval_gen):
        target_idx = Globals.TARGETS.index(target_phrase)

        outdir = os.path.join(Globals.OUTDIR, "ctc_hardl2/")
        outdir = os.path.join(outdir, "full-harmonic/")
        outdir = os.path.join(outdir, "targid_{}/".format(target_idx))
        outdir = os.path.join(outdir, "osc_{}/".format(n_osc))

        settings = {
            "indir": Globals.INDIR,
            "outdir": outdir,
            "batch_size": batch_size,
            "target": target_phrase,
            "tokens": Globals.TOKENS,
            "nsteps": Globals.NUMB_STEPS,
            "decode_step": Globals.DECODING_STEP,
            "beam_width": Globals.BEAM_WIDTH,
            "rescale": 0.95,
            "learning_rate": 250.0,
            "exp_epsilon": None,
            "gpu_device": gpu_device,
            "fn": {
                "synth": Additive.FreqHarmonic,
                "attack": attack_graph,
            },
            "synth": {
                "n_osc": n_osc,
                "initial_hz": 1e-8,
                "frame_length": 512,
                "frame_step": 512,
                "normalise": False,
            }
        }

        Boilerplate.run(settings)

        log("Finished attacking for current harmonic settings!")
    log("Finished harmonic run!")


def detnoise_inharmonic_run(gpu_device, batch_size):

    eval_gen = itertools.product(Globals.TARGETS, Globals.N_OSC)

    for test_idx, (target_phrase, n_osc) in enumerate(eval_gen):
        target_idx = Globals.TARGETS.index(target_phrase)

        outdir = os.path.join(Globals.OUTDIR, "ctc_hardl2/")
        outdir = os.path.join(outdir, "detnoise/inharmonic/")
        outdir = os.path.join(outdir, "targid_{}/".format(target_idx))
        outdir = os.path.join(outdir, "osc_{}/".format(n_osc))

        settings = {
            "indir": Globals.INDIR,
            "outdir": outdir,
            "batch_size": batch_size,
            "target": target_phrase,
            "tokens": Globals.TOKENS,
            "nsteps": int(Globals.NUMB_STEPS),
            "decode_step": Globals.DECODING_STEP,
            "beam_width": Globals.BEAM_WIDTH,
            "rescale": 0.95,
            "learning_rate": 250.0,
            "exp_epsilon": None,
            "gpu_device": gpu_device,
            "fn": {
                "synth": DeterministicPlusNoise.InharmonicPlusPlain,
                "attack": attack_graph,
            },
            "synth": {
                "n_osc": n_osc,
                "initial_hz": 1e-8,
                "frame_length": 512,
                "frame_step": 512,
                "noise_weight": 0.5,
                "normalise": False,
            },
        }

        Boilerplate.run(settings)

        log("Finished attacking for current detnoise settings!")
    log("Finished detnoise run!")


def detnoise_frequency_harmonic_run(gpu_device, batch_size):

    eval_gen = itertools.product(Globals.TARGETS, Globals.N_OSC)

    for test_idx, (target_phrase, n_osc) in enumerate(eval_gen):
        target_idx = Globals.TARGETS.index(target_phrase)

        outdir = os.path.join(Globals.OUTDIR, "ctc_hardl2/")
        outdir = os.path.join(outdir, "detnoise/freq-harmonic/")
        outdir = os.path.join(outdir, "targid_{}/".format(target_idx))
        outdir = os.path.join(outdir, "osc_{}/".format(n_osc))

        settings = {
            "indir": Globals.INDIR,
            "outdir": outdir,
            "batch_size": batch_size,
            "target": target_phrase,
            "tokens": Globals.TOKENS,
            "nsteps": int(Globals.NUMB_STEPS),
            "decode_step": Globals.DECODING_STEP,
            "beam_width": Globals.BEAM_WIDTH,
            "rescale": 0.95,
            "learning_rate": 20.0,
            "exp_epsilon": None,
            "gpu_device": gpu_device,
            "fn": {
                "synth": DeterministicPlusNoise.FreqHarmonicPlusPlain,
                "attack": attack_graph,
            },
            "synth": {
                "n_osc": n_osc,
                "initial_hz": 1e-8,
                "frame_length": 512,
                "frame_step": 512,
                "noise_weight": 0.5,
                "normalise": False,
            },
        }

        Boilerplate.run(settings)

        log("Finished attacking for current detnoise settings!")
    log("Finished detnoise run!")


def detnoise_fully_harmonic_run(gpu_device, batch_size):

    eval_gen = itertools.product(Globals.TARGETS, Globals.N_OSC)

    for test_idx, (target_phrase, n_osc) in enumerate(eval_gen):
        target_idx = Globals.TARGETS.index(target_phrase)

        outdir = os.path.join(Globals.OUTDIR, "ctc_hardl2/")
        outdir = os.path.join(outdir, "detnoise/full-harmonic/")
        outdir = os.path.join(outdir, "targid_{}/".format(target_idx))
        outdir = os.path.join(outdir, "osc_{}/".format(n_osc))

        settings = {
            "indir": Globals.INDIR,
            "outdir": outdir,
            "batch_size": batch_size,
            "target": target_phrase,
            "tokens": Globals.TOKENS,
            "nsteps": int(Globals.NUMB_STEPS),
            "decode_step": Globals.DECODING_STEP,
            "beam_width": Globals.BEAM_WIDTH,
            "rescale": 0.95,
            "learning_rate": 250.0,
            "exp_epsilon": None,
            "gpu_device": gpu_device,
            "fn": {
                "synth": DeterministicPlusNoise.FullyHarmonicPlusPlain,
                "attack": attack_graph,
            },
            "synth": {
                "n_osc": n_osc,
                "initial_hz": 1e-8,
                "frame_length": 512,
                "frame_step": 512,
                "noise_weight": 0.5,
                "normalise": False,
            },
        }

        Boilerplate.run(settings)

        log("Finished attacking for current detnoise settings!")
    log("Finished detnoise run!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "runtype",
        nargs=1,
        default=["baseline"],
        choices=[
            "all",
            "baseline",
            "inharmonic",
            "freq-harmonic",
            "full-harmonic",
            "inharmonic-det",
            "freq-harmonic-det"
            "full-harmonic-det"
        ]
    )

    parser.add_argument(
        "--gpu_device",
        nargs=1,
        type=int,
        default=[0],
        required=False
    )
    parser.add_argument(
        "--batch_size",
        nargs=1,
        type=int,
        default=[10],
        required=False
    )

    args = parser.parse_args()

    # Disable tensorflow looking at tf.app.flags.FLAGS but we get to keep
    # the args in the args variable. Otherwise we get the following exception:
    # `absl.flags._exceptions.UnrecognizedFlagError: Unknown command line flag `
    while len(sys.argv) > 1:
        sys.argv.pop()

    experiments = {
        "baseline": baseline_run,
        "inharmonic": inharmonic_run,
        "freq-harmonic": frequency_harmonic_run,
        "full-harmonic": fully_harmonic_run,
        "inharmonic-det": detnoise_inharmonic_run,
        "freq-harmonic-det": detnoise_frequency_harmonic_run,
        "full-harmonic-det": detnoise_fully_harmonic_run,
    }

    if args.runtype[0] == "all":
        for k, func in experiments.items():
            log("Running new experiment: {}".format(k))
            func(args.gpu_device[0], args.batch_size[0])

    elif args.runtype[0] in experiments.keys():
        log("Running new experiment: {}".format(args.runtype[0]))
        experiments[args.runtype[0]](args.gpu_device[0], args.batch_size[0])

    else:
        log("ERROR: {} is not a valid choice of experiment!".format(
            args.runtype[0]))
