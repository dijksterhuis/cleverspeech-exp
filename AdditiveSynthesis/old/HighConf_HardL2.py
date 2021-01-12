#!/usr/bin/env python3
import os
import sys
#import pynvml
import itertools
import argparse

from cleverspeech.Attacks import Alignments
from cleverspeech.Attacks import Graphs
from cleverspeech.Attacks import Constraints
from cleverspeech.Attacks import Losses
from cleverspeech.Attacks import Optimisers
from cleverspeech.Attacks import Procedures
from cleverspeech.Models import DeepSpeech
from cleverspeech.Synthesis import Plain, Additive, DeterministicPlusNoise
from cleverspeech.Utils import log

from cleverspeech.Experiments.AdversariesCanAdd import Boilerplate
from cleverspeech.Experiments.AdversariesCanAdd import Globals


TARGETS = [
        "open the door",
        "your cat sat on the big rug",
        "i'm glad she was able to do all the work",
        "why did it take so long for him to get into the system",
    ]

TOKENS = " abcdefghijklmnopqrstuvwxyz'-"
NUMB_STEPS = 5000
DECODING_STEP = 10
BEAM_WIDTH = 500

INDIR = "./samples"
OUTDIR = "./adv/addsynth/highconf"

#DISTANCE_LOSS_WEIGHTING = [1.0, 0.1, 0.01, 0.001]
RESCALE = 0.95
EXP_EPSILONS = [i for i in range(11, 4, -3)]
N_OSC = [4, 8, 16, 24]


# def get_gpu_memory_usage(gpu_idx=0):
#     pynvml.nvml.nvmlInit()
#     handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
#     mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#     return mem_info.free, mem_info.used, mem_info.total

def create_alignment_graphs(attack, batch):

    graph = Alignments.Graph
    alignment = graph(attack, batch)
    alignment.add_adversarial_loss(Losses.AlignmentLoss)
    alignment.create_loss_fn()
    alignment.add_optimiser(Alignments.Optimiser)

    return alignment


def attack_graph(sess, batch, synth, settings):

    attack = Graphs.SynthesisAttack(
        sess,
        batch=batch,
        constraint=Constraints.LNorm(2),
        synthesis=synth
    )
    attack.add_victim(
        DeepSpeech.Model,
        tokens=settings["tokens"]
    )
    alignment = create_alignment_graphs(attack, batch)
    attack.add_adversarial_loss(
        Losses.MaxConfidence,
        alignment.logits_alignments,
        k=5.0
    )
    attack.create_loss_fn()
    attack.add_optimiser(
        Optimisers.AdamOptimiser,
        learning_rate=settings["learning_rate"]
    )
    attack.add_procedure(
        Procedures.IterativeHardConstraintUpdate,
        alignment_graph=alignment,
        steps=settings["nsteps"],
        decode_step=settings["decode_step"],
        rescale_constant=settings["rescale"],
        exp_epsilon=settings["exp_epsilon"]
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
                "attack": attack_graph
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
            "learning_rate": 100.0,
            "exp_epsilon": None,
            "gpu_device": gpu_device,
            "fn": {
                "synth": Additive.InHarmonic,
                "attack": attack_graph
            },
            "synth": {
                "n_osc": n_osc,
                "initial_hz": 440,
                "frame_length": 512,
                "frame_step": 512,
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
            "learning_rate": 100.0,
            "exp_epsilon": None,
            "gpu_device": gpu_device,
            "fn": {
                "synth": Additive.FreqHarmonic,
                "attack": attack_graph
            },
            "synth": {
                "n_osc": n_osc,
                "initial_hz": 440,
                "frame_length": 512,
                "frame_step": 512,
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
            "learning_rate": 100.0,
            "exp_epsilon": None,
            "gpu_device": gpu_device,
            "fn": {
                "synth": Additive.FreqHarmonic,
                "attack": attack_graph
            },
            "synth": {
                "n_osc": n_osc,
                "initial_hz": 440,
                "frame_length": 512,
                "frame_step": 512,
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
            "learning_rate": 100.0,
            "exp_epsilon": None,
            "gpu_device": gpu_device,
            "fn": {
                "synth": DeterministicPlusNoise.InharmonicPlusPlain,
                "attack": attack_graph
            },
            "synth": {
                "n_osc": n_osc,
                "initial_hz": 440,
                "frame_length": 512,
                "frame_step": 512,
                "noise_weight": 0.5
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
            "learning_rate": 100.0,
            "exp_epsilon": None,
            "gpu_device": gpu_device,
            "fn": {
                "synth": DeterministicPlusNoise.FreqHarmonicPlusPlain,
                "attack": attack_graph
            },
            "synth": {
                "n_osc": n_osc,
                "initial_hz": 440,
                "frame_length": 512,
                "frame_step": 512,
                "noise_weight": 0.5
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
            "learning_rate": 100.0,
            "exp_epsilon": None,
            "gpu_device": gpu_device,
            "fn": {
                "synth": DeterministicPlusNoise.FullyHarmonicPlusPlain,
                "attack": attack_graph
            },
            "synth": {
                "n_osc": n_osc,
                "initial_hz": 440,
                "frame_length": 512,
                "frame_step": 512,
                "noise_weight": 0.5
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

    if args.runtype[0] == "all":
        baseline_run(args.gpu_device[0], args.batch_size[0])
        inharmonic_run(args.gpu_device[0], args.batch_size[0])
        frequency_harmonic_run(args.gpu_device[0], args.batch_size[0])
        fully_harmonic_run(args.gpu_device[0], args.batch_size[0])
        detnoise_inharmonic_run(args.gpu_device[0], args.batch_size[0])
        detnoise_frequency_harmonic_run(args.gpu_device[0], args.batch_size[0])
        detnoise_fully_harmonic_run(args.gpu_device[0], args.batch_size[0])

    elif args.runtype[0] == "baseline":
        baseline_run(args.gpu_device[0], args.batch_size[0])

    elif args.runtype[0] == "inharmonic":
        inharmonic_run(args.gpu_device[0], args.batch_size[0])

    elif args.runtype[0] == "freq-harmonic":
        frequency_harmonic_run(args.gpu_device[0], args.batch_size[0])

    elif args.runtype[0] == "full-harmonic":
        fully_harmonic_run(args.gpu_device[0], args.batch_size[0])

    elif args.runtype[0] == "inharmonic-det":
        detnoise_inharmonic_run(args.gpu_device[0], args.batch_size[0])

    elif args.runtype[0] == "freq-harmonic-det":
        detnoise_frequency_harmonic_run(args.gpu_device[0], args.batch_size[0])

    elif args.runtype[0] == "freq-harmonic-det":
        detnoise_fully_harmonic_run(args.gpu_device[0], args.batch_size[0])

    else:
        baseline_run(args.gpu_device[0], args.batch_size[0])



