#!/usr/bin/env python3
import os

from cleverspeech import data
from cleverspeech import graph
from cleverspeech.utils.Utils import log
from cleverspeech.utils.runtime.ExperimentArguments import args
from cleverspeech.utils.runtime.Execution import default_manager
from cleverspeech.data.ingress.etl.batch_generators import PATH_GENERATORS


# victim model
from SecEval import VictimAPI as DeepSpeech


LOSS_CHOICES = {
    "softmax": graph.Losses.BiggioMaxMinSoftmax,
    "logits": graph.Losses.BiggioMaxMin,
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

    attack.add_loss(
        LOSS_CHOICES[settings["loss"]],
        attack.placeholders.targets,
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
    loss = master_settings["loss"]
    decoder = master_settings["decoder"]
    outdir = master_settings["outdir"]

    attack_type = os.path.basename(__file__).replace(".py", "")

    outdir = os.path.join(outdir, attack_type)
    outdir = os.path.join(outdir, "baselines/biggio/")
    outdir = os.path.join(outdir, "{}/".format(align))
    outdir = os.path.join(outdir, "{}/".format(loss))
    outdir = os.path.join(outdir, "{}/".format(decoder))

    master_settings["outdir"] = outdir

    batch_gen = data.ingress.etl.batch_generators.PATH_GENERATORS[align](master_settings)

    default_manager(
        master_settings,
        create_attack_graph,
        batch_gen,
    )
    log("Finished run.")


if __name__ == '__main__':
    log("", wrap=True)

    extra_args = {
        'loss': [str, "logits", False, LOSS_CHOICES.keys()],
    }

    args(attack_run, additional_args=extra_args)



