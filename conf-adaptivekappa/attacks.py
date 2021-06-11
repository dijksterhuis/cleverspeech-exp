#!/usr/bin/env python3
import os

from cleverspeech import data
from cleverspeech import graph
from cleverspeech.utils.Utils import log
from cleverspeech.utils.runtime.Execution import default_evasion_manager
from cleverspeech.utils.runtime.ExperimentArguments import args
from cleverspeech.graph.CTCAlignmentSearch import create_tf_ctc_alignment_search_graph

# victim model import
from SecEval import VictimAPI as DeepSpeech


ALIGNMENT_CHOICES = {
    "sparse": data.ingress.etl.batch_generators.sparse,
    "mid": data.ingress.etl.batch_generators.midish,
    "dense": data.ingress.etl.batch_generators.dense,
    "ctcalign": data.ingress.etl.batch_generators.standard,
}


def create_attack_graph(sess, batch, settings):

    feeds = data.ingress.Feeds.Attack(batch)

    attack = graph.AttackConstructors.EvasionAttackConstructor(sess, batch, feeds)
    attack.add_placeholders(
        graph.Placeholders.Placeholders
    )
    attack.add_hard_constraint(
        graph.Constraints.L2,
        r_constant=settings["rescale"],
        update_method=settings["constraint_update"],
    )
    attack.add_perturbation_subgraph(
        graph.PerturbationSubGraphs.Independent
    )
    attack.add_victim(
        DeepSpeech.Model,
        decoder=settings["decoder"],
        beam_width=settings["beam_width"]
    )

    if settings["align"] == "ctcalign":

        alignment = create_tf_ctc_alignment_search_graph(sess, batch)

        attack.add_loss(
            graph.Losses.AdaptiveKappaMaxDiff,
            alignment.graph.target_alignments,
            k=settings["kappa"]
        )
        attack.create_loss_fn()
        attack.add_optimiser(
            graph.Optimisers.AdamIndependentOptimiser,
            learning_rate=settings["learning_rate"]
        )
        attack.add_procedure(
            graph.Procedures.StandardCTCAlignProcedure,
            alignment,
            steps=settings["nsteps"],
            update_step=settings["decode_step"],
            loss_update_idx=[0],
        )

    else:
        attack.add_loss(
            graph.Losses.AdaptiveKappaMaxDiff,
            attack.placeholders.targets,
            k=settings["kappa"]
        )
        attack.create_loss_fn()
        attack.add_optimiser(
            graph.Optimisers.AdamIndependentOptimiser,
            learning_rate=settings["learning_rate"]
        )
        attack.add_procedure(
            graph.Procedures.StandardProcedure,
            steps=settings["nsteps"],
            update_step=settings["decode_step"],
            loss_update_idx=[0],
        )

    return attack


def attack_run(master_settings):
    """
    Special variant of Carlini & Wagner's improved loss function from the
    original audio paper, with kappa as a vector of frame-wise differences
    between max(other_classes) and min(other_classes).

    :param master_settings: a dictionary of arguments to run the attack, as
    defined by command line arguments. Will override the settings dictionary
    defined below.

    :return: None
    """

    align = master_settings["align"]
    decoder = master_settings["decoder"]
    kappa = master_settings["kappa"]
    outdir = master_settings["outdir"]

    outdir = os.path.join(outdir, "evasion/confidence/adaptive-kappa/")
    outdir = os.path.join(outdir, "{}/".format(align))
    outdir = os.path.join(outdir, "{}/".format(decoder))
    outdir = os.path.join(outdir, "{}/".format(kappa))

    master_settings["outdir"] = outdir

    batch_gen = ALIGNMENT_CHOICES[align](master_settings)

    default_evasion_manager(
        master_settings,
        create_attack_graph,
        batch_gen,
    )
    log("Finished run.")


if __name__ == '__main__':

    extra_args = {
        'align': [str, "sparse", False, ALIGNMENT_CHOICES.keys()],
        "kappa": [float, 0.25, False, None],
    }

    args(attack_run, extra_args)

