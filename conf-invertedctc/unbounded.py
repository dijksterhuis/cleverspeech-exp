#!/usr/bin/env python3
import os
import traceback
import multiprocessing as mp

from cleverspeech import data
from cleverspeech import graph
from cleverspeech.utils.Utils import log
from cleverspeech.utils.runtime.Execution import default_unbounded_manager
from cleverspeech.utils.runtime.ExperimentArguments import args
from cleverspeech.graph.CTCAlignmentSearch import create_tf_ctc_alignment_search_graph

# Victim model import
from SecEval import VictimAPI as DeepSpeech


ALIGNMENT_CHOICES = {
    "sparse": data.ingress.etl.batch_generators.sparse,
    "mid": data.ingress.etl.batch_generators.midish,
    "dense": data.ingress.etl.batch_generators.dense,
    "ctcalign": data.ingress.etl.batch_generators.standard,
}


def create_attack_graph(sess, batch, settings):

    feeds = data.ingress.Feeds.Attack(batch)

    attack = graph.AttackConstructors.UnboundedAttackConstructor(sess, batch, feeds)
    attack.add_placeholders(graph.Placeholders.Placeholders)
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
            graph.Losses.GreedyOtherAlignmentsCTCLoss,
            alignment=alignment.graph.target_alignments,
            weight_settings=(1/100, 1/100)
        )
        attack.add_loss(
            graph.Losses.CWMaxDiff,
            alignment.graph.target_alignments,
            k=settings["kappa"]
        )
        attack.create_loss_fn()
        attack.add_optimiser(
            graph.Optimisers.AdamIndependentOptimiser,
            learning_rate=settings["learning_rate"]
        )
        attack.add_procedure(
            graph.Procedures.CTCAlignUnbounded,
            alignment_graph=alignment,
            steps=settings["nsteps"],
            update_step=settings["decode_step"]
        )

    else:

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
            graph.Procedures.Unbounded,
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

    outdir = os.path.join(outdir, "unbounded/confidence/invertedctc-cwmaxdiff/")
    outdir = os.path.join(outdir, "{}/".format(align))
    outdir = os.path.join(outdir, "{}/".format(decoder))
    outdir = os.path.join(outdir, "{}/".format(kappa))

    master_settings["outdir"] = outdir

    batch_gen = ALIGNMENT_CHOICES[align](master_settings)

    default_unbounded_manager(
        master_settings,
        create_attack_graph,
        batch_gen,
    )
    log("Finished run.")


if __name__ == '__main__':

    extra_args = {
        'align': [str, "sparse", False, ALIGNMENT_CHOICES.keys()],
        "kappa": [float, 5.0, False, None],
    }

    args(attack_run, additional_args=extra_args)

