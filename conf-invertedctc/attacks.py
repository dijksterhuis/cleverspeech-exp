#!/usr/bin/env python3
import os

from cleverspeech import data
from cleverspeech import graph
from cleverspeech.utils.Utils import log
from cleverspeech.utils.runtime.Execution import default_evasion_manager
from cleverspeech.utils.runtime.ExperimentArguments import args


# Victim model import
from SecEval import VictimAPI as DeepSpeech


def create_attack_graph(sess, batch, settings):

    feeds = data.ingress.Feeds.Attack(batch)

    attack = graph.AttackConstructors.EvasionAttackConstructor(sess, batch, feeds)
    attack.add_placeholders(graph.Placeholders.Placeholders)
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
    attack.add_loss(
        graph.Losses.GreedyOtherAlignmentsCTCLoss,
        alignment=attack.placeholders.targets,
        weight_settings=(1 / 100, 1 / 100)
    )
    attack.add_loss(
        graph.Losses.CWMaxDiff,
        attack.placeholders.targets,
        k=settings["kappa"]
    )
    attack.create_loss_fn()
    attack.add_optimiser(
        graph.Optimisers.AdamIndependentOptimiser,
        learning_rate=settings["learning_rate"]
    )
    attack.add_procedure(
        graph.Procedures.EvasionPGD,
        steps=settings["nsteps"],
        update_step=settings["decode_step"]
    )

    return attack


def attack_run(master_settings):
    """
    """

    align = master_settings["align"]
    decoder = master_settings["decoder"]
    kappa = master_settings["kappa"]
    outdir = master_settings["outdir"]

    outdir = os.path.join(outdir, "evasion/confidence/invertedctc-cwmaxdiff/")
    outdir = os.path.join(outdir, "{}/".format(align))
    outdir = os.path.join(outdir, "{}/".format(decoder))
    outdir = os.path.join(outdir, "{}/".format(kappa))

    master_settings["outdir"] = outdir

    batch_gen = data.ingress.etl.batch_generators.PATH_GENERATORS[align](master_settings)

    default_evasion_manager(
        master_settings,
        create_attack_graph,
        batch_gen,
    )
    log("Finished run.")


if __name__ == '__main__':

    extra_args = {
        "kappa": [float, 5.0, False, None],
    }

    args(attack_run, additional_args=extra_args)

