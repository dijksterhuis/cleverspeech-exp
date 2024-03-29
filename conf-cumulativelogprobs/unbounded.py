#!/usr/bin/env python3
import os

from cleverspeech import data
from cleverspeech import graph
from cleverspeech.utils.Utils import log
from cleverspeech.utils.runtime.Execution import manager
from cleverspeech.utils.runtime.ExperimentArguments import args


# victim model
from SecEval import VictimAPI as DeepSpeech

# local attack classes
import custom_defs


LOSS_CHOICES = {
    "fwd": custom_defs.FwdOnlyLogProbsLoss,
    "back": custom_defs.BackOnlyLogProbsLoss,
    "fwdplusback": custom_defs.FwdPlusBackLogProbsLoss,
    "fwdmultback": custom_defs.FwdMultBackLogProbsLoss,
}


def create_attack_graph(sess, batch, settings):

    attack = graph.AttackConstructors.UnboundedAttackConstructor(
        sess, batch
    )
    attack.add_path_search(
        graph.Paths.ALL_PATHS[settings["align"]]
    )
    attack.add_placeholders(
        graph.Placeholders.Placeholders
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
    )
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


def custom_extract_results(attack):

    results = data.egress.extract.get_attack_state(attack)

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
    outdir = master_settings["outdir"]

    attack_type = os.path.basename(__file__).replace(".py", "")

    outdir = os.path.join(outdir, attack_type)
    outdir = os.path.join(outdir, "confidence/cumulative_logprobs/")
    outdir = os.path.join(outdir, "{}/".format(align))
    outdir = os.path.join(outdir, "{}/".format(decoder))
    outdir = os.path.join(outdir, "{}/".format(loss))

    master_settings["outdir"] = outdir

    batch_gen = data.ingress.mcv_v1.BatchIterator(master_settings)

    manager(
        master_settings,
        create_attack_graph,
        batch_gen,
        results_extract_fn=custom_extract_results,
    )
    log("Finished run.")


if __name__ == '__main__':

    extra_args = {
        "loss": [str, "fwd", False, LOSS_CHOICES.keys()],
    }

    args(attack_run, additional_args=extra_args)

