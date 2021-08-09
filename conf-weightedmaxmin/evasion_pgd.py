#!/usr/bin/env python3
import os
import tensorflow as tf

from cleverspeech import data
from cleverspeech import graph
from cleverspeech.utils.Utils import log
from cleverspeech.utils.runtime.Execution import manager
from cleverspeech.utils.runtime.ExperimentArguments import args



# victim model
from SecEval import VictimAPI as DeepSpeech


class CustomLoss(graph.Losses.BaseLogitDiffLoss):
    def __init__(self, attack, weight_settings=(1.0, 1.0)):

        super().__init__(
            attack,
            softmax=True,
            weight_settings=weight_settings,
        )

        self.c = c = 2 - self.target_logit
        diff = c * (- self.target_logit + self.max_other_logit)

        self.loss_fn = tf.reduce_sum(diff, axis=1)
        self.loss_fn += attack.batch.audios["max_feats"]  # loss = 0 when min'd
        self.loss_fn *= self.weights


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
        CustomLoss,
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


def custom_extract_results(attack):
    results = data.egress.extract.get_attack_state(attack)
    results["loss_weightings"] = attack.procedure.tf_run(
        attack.loss[0].c
    )
    return results


def attack_run(master_settings):

    align = master_settings["align"]
    decoder = master_settings["decoder"]
    outdir = master_settings["outdir"]

    attack_type = os.path.basename(__file__).replace(".py", "")

    outdir = os.path.join(outdir, attack_type)
    outdir = os.path.join(outdir, "confidence/weightedmaxmin/")
    outdir = os.path.join(outdir, "{}/".format(align))
    outdir = os.path.join(outdir, "{}/".format(decoder))

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

    log("", wrap=True)

    extra_args = {
        }

    args(attack_run, additional_args=extra_args)

