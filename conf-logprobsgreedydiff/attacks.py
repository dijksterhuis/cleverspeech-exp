#!/usr/bin/env python3
import os

from collections import OrderedDict

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


ALIGNMENT_CHOICES = {
    "sparse": batch_generators.sparse,
    "ctcalign": batch_generators.standard,
    "dense": batch_generators.dense,
}

LOSS_CHOICES = {
    "fwd": custom_defs.FwdOnlyVibertish,
    "back": custom_defs.BackOnlyVibertish,
    "fwdplusback": custom_defs.FwdPlusBackVibertish,
    "fwdmultback": custom_defs.FwdMultBackVibertish,
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
        decoder=settings["decoder"],
        beam_width=settings["beam_width"]
    )

    if settings["align"] == "ctcalign":

        alignment = create_tf_ctc_alignment_search_graph(sess, batch)

        attack.add_loss(
            LOSS_CHOICES[settings["loss"]],
            alignment.graph.target_alignments,
            kappa=settings["kappa"],
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
            LOSS_CHOICES[settings["loss"]],
            attack.placeholders.targets,
            kappa=settings["kappa"],
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
    kappa = master_settings["kappa"]
    outdir = master_settings["outdir"]

    outdir = os.path.join(outdir, "evasion/confidence/logprobs-greedydiff/")
    outdir = os.path.join(outdir, "{}/".format(align))
    outdir = os.path.join(outdir, "{}/".format(decoder))
    outdir = os.path.join(outdir, "{}/".format(loss))
    outdir = os.path.join(outdir, "{}/".format(kappa))
    master_settings["outdir"] = outdir

    batch_gen = ALIGNMENT_CHOICES[align](master_settings)

    execute(master_settings, create_attack_graph, batch_gen,)
    log("Finished run.")


if __name__ == '__main__':

    extra_args = {
        'align': [str, "sparse", False, ALIGNMENT_CHOICES.keys()],
        "loss": [str, "fwd", False, LOSS_CHOICES.keys()],
        "kappa": [float, 2.0, False, None],
    }

    args(attack_run, additional_args=extra_args)
