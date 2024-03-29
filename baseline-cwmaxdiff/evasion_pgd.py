#!/usr/bin/env python3
import os
from cleverspeech import data
from cleverspeech import graph
from cleverspeech.utils.Utils import log
from cleverspeech.utils.runtime.Execution import default_manager
from cleverspeech.utils.runtime.ExperimentArguments import args

# attack def imports


# victim model
from SecEval import VictimAPI as DeepSpeech


LOSS_CHOICES = {
    "softmax": graph.Losses.CWMaxDiffSoftmax,
    "logits": graph.Losses.CWMaxDiff,
}


def create_attack_graph(sess, batch, settings):

    attack = graph.AttackConstructors.EvasionAttackConstructor(
        sess, batch
    )
    attack.add_path_search(
        graph.Paths.ALL_PATHS[settings["align"]]
    )
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
    attack.add_loss(
        LOSS_CHOICES[settings["loss"]],
        k=settings["kappa"]
    )
    attack.add_optimiser(
        graph.Optimisers.AdamIndependentOptimiser,
        learning_rate=settings["learning_rate"]
    )
    attack.add_procedure(
        graph.Procedures.EvasionCGD,
        steps=settings["nsteps"],
        update_step=settings["decode_step"]
    )

    return attack


def attack_run(master_settings):
    """
    Use Carlini & Wagner's improved loss function form the original audio paper,
    but reintroduce kappa from the image attack as we're looking to perform
    targeted maximum-confidence evasion attacks --- i.e. not just find minimum
    perturbations.

    :param master_settings: a dictionary of arguments to run the attack, as
    defined by command line arguments. Will override the settings dictionary
    defined below.

    :return: None
    """

    align = master_settings["align"]
    loss = master_settings["loss"]
    decoder = master_settings["decoder"]
    kappa = master_settings["kappa"]
    outdir = master_settings["outdir"]

    attack_type = os.path.basename(__file__).replace(".py", "")

    outdir = os.path.join(outdir, attack_type)
    outdir = os.path.join(outdir, "baselines/cwmaxdiff/")
    outdir = os.path.join(outdir, "{}/".format(align))
    outdir = os.path.join(outdir, "{}/".format(loss))
    outdir = os.path.join(outdir, "{}/".format(decoder))
    outdir = os.path.join(outdir, "{}/".format(kappa))

    master_settings["outdir"] = outdir
    master_settings["attack type"] = attack_type

    batch_gen = data.ingress.mcv_v1.BatchIterator(master_settings)

    default_manager(
        master_settings,
        create_attack_graph,
        batch_gen,
    )
    log("Finished run.")


if __name__ == '__main__':

    log("", wrap=True)

    extra_args = {
        "kappa": [float, 0.0, False, None],
        'loss': [str, "logits", False, LOSS_CHOICES.keys()],
    }

    if extra_args["loss"][1] == "softmax":
        assert 0 <= extra_args["kappa"][1] < 1

    else:
        assert extra_args["kappa"][1] >= 0

    args(attack_run, additional_args=extra_args)



