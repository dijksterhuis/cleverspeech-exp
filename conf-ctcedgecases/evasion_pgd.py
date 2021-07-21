#!/usr/bin/env python3
import os

from cleverspeech import data
from cleverspeech import graph
from cleverspeech.utils.Utils import log
from cleverspeech.utils.runtime.Execution import default_manager
from cleverspeech.utils.runtime.ExperimentArguments import args



# victim model
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

    if settings["procedure"] == "std":

        attack.add_loss(
            graph.Losses.AlignmentsCTCLoss
        )
        attack.create_loss_fn()
        attack.add_optimiser(
            graph.Optimisers.AdamIndependentOptimiser,
            learning_rate=settings["learning_rate"]
        )
        attack.add_procedure(
            graph.Procedures.EvasionCGD,
            steps=settings["nsteps"],
            update_step=settings["decode_step"],
        )

    elif settings["procedure"] == "extreme":

        attack.add_loss(
            graph.Losses.AlignmentsCTCLoss
        )
        attack.create_loss_fn()
        attack.add_optimiser(
            graph.Optimisers.AdamIndependentOptimiser,
            learning_rate=settings["learning_rate"]
        )
        attack.add_procedure(
            graph.Procedures.LossCGD,
            steps=settings["nsteps"],
            update_step=settings["decode_step"],
            loss_lower_bound=settings["loss_threshold"],
        )

    else:
        raise NotImplementedError

    return attack


def attack_run(master_settings):
    """

    """

    align = master_settings["align"]
    decoder = master_settings["decoder"]
    procedure = master_settings["procedure"]
    loss_threshold = master_settings["loss_threshold"]
    outdir = master_settings["outdir"]

    attack_type = os.path.basename(__file__).replace(".py", "")

    outdir = os.path.join(outdir, attack_type)
    outdir = os.path.join(outdir, "confidence/ctc-edge-case/")
    outdir = os.path.join(outdir, "{}/".format(align))
    outdir = os.path.join(outdir, "{}/".format(decoder))
    outdir = os.path.join(outdir, "{}/".format(procedure))

    if procedure == "extreme":
        outdir = os.path.join(outdir, "{}/".format(loss_threshold))

    master_settings["outdir"] = outdir

    batch_gen = data.ingress.etl.batch_generators.PATH_GENERATORS[align](master_settings)

    default_manager(
        master_settings,
        create_attack_graph,
        batch_gen,
    )
    log("Finished run.")


if __name__ == '__main__':

    extra_args = {
        "procedure": [str, "std", False, ["std", "extreme"]],
        "loss_threshold": [float, 20.0, False, None],
    }

    args(attack_run, additional_args=extra_args)

