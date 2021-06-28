#!/usr/bin/env python3
import os

from cleverspeech import data
from cleverspeech import graph
from cleverspeech.utils.Utils import log
from cleverspeech.utils.runtime.Execution import manager
from cleverspeech.utils.runtime.ExperimentArguments import args


# victim model import
from SecEval import VictimAPI as DeepSpeech

# local attack classes
import custom_defs


LOSS_CHOICES = {
    "fwd": custom_defs.FwdOnlyVibertish,
    "back": custom_defs.BackOnlyVibertish,
    "fwdplusback": custom_defs.FwdPlusBackVibertish,
    "fwdmultback": custom_defs.FwdMultBackVibertish,
}


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
        LOSS_CHOICES[settings["loss"]],
        attack.placeholders.targets,
        kappa=settings["kappa"],
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


def custom_extract_results(attack):

    results = data.egress.extract.get_evasion_attack_state(attack)

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

    batch_gen = data.ingress.etl.batch_generators.PATH_GENERATORS[align](master_settings)

    manager(
        master_settings,
        create_attack_graph,
        batch_gen,
        results_extract_fn=custom_extract_results,
        results_transform_fn=data.egress.transform.evasion_gen,
    )
    log("Finished run.")


if __name__ == '__main__':

    extra_args = {
        "loss": [str, "fwd", False, LOSS_CHOICES.keys()],
        "kappa": [float, 2.0, False, None],
    }

    args(attack_run, additional_args=extra_args)
