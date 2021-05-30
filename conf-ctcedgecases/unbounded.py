#!/usr/bin/env python3
import os

from cleverspeech import data
from cleverspeech import graph
from cleverspeech.utils.Utils import log
from cleverspeech.utils.runtime.Execution import default_unbounded_manager
from cleverspeech.utils.runtime.ExperimentArguments import args
from cleverspeech.graph.CTCAlignmentSearch import create_tf_ctc_alignment_search_graph

# victim model
from SecEval import VictimAPI as DeepSpeech


ALIGNMENT_CHOICES = {
    "sparse": data.ingress.etl.batch_generators.sparse,
    "ctcalign": data.ingress.etl.batch_generators.standard,
    "dense": data.ingress.etl.batch_generators.dense,
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
            graph.Losses.AlignmentsCTCLoss,
            alignment=alignment.graph.target_alignments,
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
            update_step=settings["decode_step"],
        )

    else:

        attack.add_loss(
            graph.Losses.AlignmentsCTCLoss
        )
        attack.create_loss_fn()
        attack.add_optimiser(
            graph.Optimisers.AdamIndependentOptimiser,
            learning_rate=settings["learning_rate"]
        )
        attack.add_procedure(
            graph.Procedures.Unbounded,
            steps=settings["nsteps"],
            update_step=settings["decode_step"],
        )

    return attack


def attack_run(master_settings):
    """

    """

    align = master_settings["align"]
    decoder = master_settings["decoder"]
    procedure = master_settings["procedure"]
    loss_threshold = master_settings["loss_threshold"]
    outdir = master_settings["outdir"]

    outdir = os.path.join(outdir, "unbounded/confidence/ctc-edge-case/")
    outdir = os.path.join(outdir, "{}/".format(align))
    outdir = os.path.join(outdir, "{}/".format(decoder))
    outdir = os.path.join(outdir, "{}/".format(procedure))

    if procedure == "extreme":
        outdir = os.path.join(outdir, "{}/".format(loss_threshold))

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
        "procedure": [str, "std", False, ["std", "extreme"]],
        "loss_threshold": [float, 20.0, False, None],
    }

    args(attack_run, additional_args=extra_args)

