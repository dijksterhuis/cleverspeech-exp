#!/usr/bin/env python3
import os

from cleverspeech.graph.AttackConstructors import EvasionAttackConstructor
from cleverspeech.graph import Constraints
from cleverspeech.graph import PerturbationSubGraphs
from cleverspeech.graph import Losses
from cleverspeech.graph import Optimisers
from cleverspeech.graph import Procedures
from cleverspeech.graph import Placeholders
from cleverspeech.graph.CTCAlignmentSearch import create_tf_ctc_alignment_search_graph

from cleverspeech.data.ingress.etl import batch_generators
from cleverspeech.data.ingress import Feeds
from cleverspeech.data.egress.Databases import SingleJsonDB
from cleverspeech.data.egress import AttackETLs
from cleverspeech.data.egress.Writers import SingleFileWriter
from cleverspeech.data.egress import Reporting

from cleverspeech.utils.runtime.AttackSpawner import AttackSpawner
from cleverspeech.utils.runtime.ExperimentArguments import args
from cleverspeech.utils.Utils import log, lcomp

# victim model import
from SecEval import VictimAPI as DeepSpeech

# local attack classes
import custom_defs

GPU_DEVICE = 0
MAX_PROCESSES = 1
SPAWN_DELAY = 30

AUDIOS_INDIR = "./samples/all/"
TARGETS_PATH = "./samples/cv-valid-test.csv"
OUTDIR = "./adv/"
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

LOSSES = {
    "fwd": custom_defs.FwdOnlyLogProbsLoss,
    "back": custom_defs.BackOnlyLogProbsLoss,
    "fwdplusback": custom_defs.FwdPlusBackLogProbsLoss,
    "fwdmultback": custom_defs.FwdMultBackLogProbsLoss,
}

# VIBERT-ish
# ==============================================================================
# Main idea: We should optimise a specific alignment to become more likely than
# all others instead of optimising for individual class labels per frame.


def mod_convert_evasion_attack_state_to_dict(attack):

    results = AttackETLs.convert_evasion_attack_state_to_dict(attack)

    target_alpha = attack.loss[0].fwd_target_log_probs
    target_beta = attack.loss[0].back_target_log_probs

    alpha, beta = attack.procedure.tf_run(
        [target_alpha, target_beta]
    )

    results.update(
        {
            "alpha": alpha,
            "beta": beta,
        }
    )

    return results


def execute(settings, attack_fn, batch_gen):

    # set up the directory we'll use for results

    if not os.path.exists(settings["outdir"]):
        os.makedirs(settings["outdir"], exist_ok=True)

    results_extractor = mod_convert_evasion_attack_state_to_dict
    results_transformer = AttackETLs.EvasionResults()

    file_writer = SingleFileWriter(settings["outdir"], results_transformer)

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
            attack_args = (settings, attack_fn, batch, results_extractor)
            spawner.spawn(attack_args)

    # Run the stats function on all successful examples once all attacks
    # are completed.
    Reporting.generate_stats_file(settings["outdir"])


def create_attack_graph(sess, batch, settings):

    feeds = Feeds.Attack(batch)

    attack = EvasionAttackConstructor(sess, batch, feeds)
    attack.add_placeholders(Placeholders.Placeholders)
    attack.add_hard_constraint(
        Constraints.L2,
        r_constant=settings["rescale"],
        update_method=settings["constraint_update"],
    )
    attack.add_perturbation_subgraph(
        PerturbationSubGraphs.Independent
    )
    attack.add_victim(
        DeepSpeech.Model,
        tokens=settings["tokens"],
        decoder=settings["decoder_type"],
        beam_width=settings["beam_width"]
    )

    if settings["align"] == "ctcalign":

        alignment = create_tf_ctc_alignment_search_graph(sess, batch)

        attack.add_loss(
            LOSSES[settings["loss_type"]],
            alignment.graph.target_alignments,
        )
        attack.create_loss_fn()
        attack.add_optimiser(
            Optimisers.AdamIndependentOptimiser,
            learning_rate=settings["learning_rate"]
        )
        attack.add_procedure(
            Procedures.StandardCTCAlignProcedure,
            alignment,
            steps=settings["nsteps"],
            update_step=settings["decode_step"],
        )

    else:

        attack.add_loss(
            LOSSES[settings["loss_type"]],
            attack.placeholders.targets,
        )
        attack.create_loss_fn()
        attack.add_optimiser(
            Optimisers.AdamIndependentOptimiser,
            learning_rate=settings["learning_rate"]
        )
        attack.add_procedure(
            Procedures.StandardProcedure,
            steps=settings["nsteps"],
            update_step=settings["decode_step"]
        )

    return attack


def attack_run(master_settings):

    align = master_settings["align"]
    decoder = master_settings["decoder"]
    loss = master_settings["loss"]

    outdir = os.path.join(OUTDIR, "evasion/confidence/cumulative_logprobs/")
    outdir = os.path.join(outdir, "{}/".format(align))
    outdir = os.path.join(outdir, "{}/".format(decoder))
    outdir = os.path.join(outdir, "{}/".format(loss))

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
        "align": align,
        "decoder_type": decoder,
        "loss_type": loss,
    }

    settings.update(master_settings)

    if align == "ctcalign":
        batch_gen = batch_generators.standard(settings)

    elif align == "dense":
        batch_gen = batch_generators.dense(settings)

    elif align == "sparse":
        batch_gen = batch_generators.sparse(settings)

    else:
        raise NotImplementedError

    execute(settings, create_attack_graph, batch_gen)
    log("Finished run.")


if __name__ == '__main__':

    extra_args = {
        'align': [str, "sparse", False, ["sparse", "ctcalign", "dense"]],
        'decoder': [str, "batch", False, ["greedy", "batch", "ds", "tf"]],
        "loss": [str, "fwd", False, ["fwd", "back", "fwdplusback", "fwdmultback"]],
    }

    args(attack_run, additional_args=extra_args)

