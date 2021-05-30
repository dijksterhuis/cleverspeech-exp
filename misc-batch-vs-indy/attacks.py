#!/usr/bin/env python3
import os

from cleverspeech import data
from cleverspeech import graph
from cleverspeech.utils.Utils import log
from cleverspeech.utils.runtime.Execution import default_evasion_manager
from cleverspeech.utils.runtime.ExperimentArguments import args

# victim model
from SecEval import VictimAPI as DeepSpeech


GRAPH_CHOICES = {
    "batch": [
        graph.PerturbationSubGraphs.Batch,
        graph.Optimisers.AdamBatchwiseOptimiser,
    ],
    "indy": [
        graph.PerturbationSubGraphs.Independent,
        graph.Optimisers.AdamIndependentOptimiser,
    ],
}

LOSS_CHOICES = {
    "ctc": graph.Losses.CTCLoss,
    "ctc2": graph.Losses.CTCLossV2
}


def create_attack_graph(sess, batch, settings):

    perturbation_sub_graph_cls, optimiser_cls = GRAPH_CHOICES[settings["graph"]]

    feeds = data.ingress.Feeds.Attack(batch)

    attack = graph.AttackConstructors.EvasionAttackConstructor(sess, batch, feeds)
    attack.add_placeholders(graph.Placeholders.Placeholders)
    attack.add_hard_constraint(
        graph.Constraints.L2,
        r_constant=settings["rescale"],
        update_method=settings["constraint_update"],
    )
    attack.add_perturbation_subgraph(
        perturbation_sub_graph_cls
    )
    attack.add_victim(
        DeepSpeech.Model,
        decoder=settings["decoder"],
        beam_width=settings["beam_width"]
    )
    attack.add_loss(
        LOSS_CHOICES[settings["loss"]]
    )
    attack.create_loss_fn()
    attack.add_optimiser(
        optimiser_cls,
        learning_rate=settings["learning_rate"]
    )
    attack.add_procedure(
        graph.Procedures.StandardProcedure,
        steps=settings["nsteps"],
        update_step=settings["decode_step"]
    )

    return attack


def attack_run(master_settings):

    graph_type = master_settings["graph"]
    decoder = master_settings["decoder"]
    loss = master_settings["loss"]
    nbatch_max = master_settings["nbatch_max"]
    nbatch_step = master_settings["nbatch_step"]
    initial_outdir = master_settings["outdir"]

    assert nbatch_max >= 1
    assert nbatch_step >= 1
    assert nbatch_max >= nbatch_step

    for batch_size in range(0, nbatch_max + 1, nbatch_step):

        if batch_size == 0:
            batch_size = 1

        outdir = os.path.join(initial_outdir, "evasion/batch-vs-indy/")
        outdir = os.path.join(outdir, "{}/".format(graph_type))
        outdir = os.path.join(outdir, "{}/".format(decoder))
        outdir = os.path.join(outdir, "{}/".format(loss))
        outdir = os.path.join(outdir, "{}/".format(batch_size))

        master_settings["outdir"] = outdir
        master_settings["batch_size"] = batch_size
        master_settings["max_examples"] = batch_size

        batch_gen = data.ingress.etl.batch_generators.standard(master_settings)
        default_evasion_manager(
            master_settings,
            create_attack_graph,
            batch_gen,
        )

        log("Finished batch run {}.".format(batch_size))

    log("Finished all runs.")


if __name__ == '__main__':

    log("", wrap=True)

    extra_args = {
        'graph': [str, "batch", False, GRAPH_CHOICES.keys()],
        'loss': [str, "ctc", False, LOSS_CHOICES.keys()],
        'nbatch_max': [int, 20, False, None],
        'nbatch_step': [int, 5, False, None],
    }

    args(attack_run, additional_args=extra_args)



