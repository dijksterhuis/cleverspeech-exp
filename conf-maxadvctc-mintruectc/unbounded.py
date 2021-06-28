#!/usr/bin/env python3
import os

from cleverspeech import data
from cleverspeech import graph
from cleverspeech.utils.Utils import log
from cleverspeech.utils.runtime.Execution import default_unbounded_manager
from cleverspeech.utils.runtime.ExperimentArguments import args


# victim model
from SecEval import VictimAPI as DeepSpeech

# custom defs
from custom_defs import OtherTranscriptionCTCLoss, TruePlaceholders, TrueFeeds


LOSS_CHOICES = {
    "ctc": graph.Losses.CTCLoss,
    "ctc2": graph.Losses.CTCLossV2,
}


def create_attack_graph(sess, batch, settings):

    feeds = TrueFeeds(batch)

    attack = graph.AttackConstructors.UnboundedAttackConstructor(sess, batch, feeds)
    attack.add_placeholders(
        TruePlaceholders
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
    attack.add_loss(
        LOSS_CHOICES[settings["loss"]]
    )
    attack.add_loss(
        OtherTranscriptionCTCLoss,
        weight_settings=(-0.1, -1.0)
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

    loss = master_settings["loss"]
    outdir = master_settings["outdir"]

    attack_type = os.path.basename(__file__).replace(".py", "")

    outdir = os.path.join(outdir, attack_type)
    outdir = os.path.join(outdir, "confidence/maxctc-mintruectc/")
    outdir = os.path.join(outdir, "{}/".format(loss))

    master_settings["outdir"] = outdir

    batch_gen = data.ingress.etl.batch_generators.standard(master_settings)
    default_unbounded_manager(
        master_settings,
        create_attack_graph,
        batch_gen,
    )

    log("Finished run.")  # {}.".format(run))


if __name__ == '__main__':

    log("", wrap=True)

    extra_args = {
        "loss": [str, "ctc", False, LOSS_CHOICES.keys()],
    }

    args(attack_run, additional_args=extra_args)



