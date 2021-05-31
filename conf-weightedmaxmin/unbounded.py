#!/usr/bin/env python3
import os
import tensorflow as tf

from cleverspeech import data
from cleverspeech import graph
from cleverspeech.utils.Utils import log
from cleverspeech.utils.runtime.Execution import manager

from cleverspeech.utils.runtime.ExperimentArguments import args
from cleverspeech.graph.CTCAlignmentSearch import create_tf_ctc_alignment_search_graph

# victim model
from SecEval import VictimAPI as DeepSpeech


ALIGNMENT_CHOICES = {
    "sparse": data.ingress.etl.batch_generators.sparse,
    "ctcalign": data.ingress.etl.batch_generators.standard,
    "dense": data.ingress.etl.batch_generators.dense,
}


class CustomLoss(graph.Losses.BaseLogitDiffLoss):
    def __init__(self, attack, target_logits, weight_settings=(1.0, 1.0)):

        super().__init__(
            attack,
            target_logits,
            softmax=True,
            weight_settings=weight_settings,
        )

        self.c = c = 2 - self.target_logit
        diff = c * (- self.target_logit + self.max_other_logit)

        self.loss_fn = tf.reduce_sum(diff, axis=1)
        self.loss_fn += attack.batch.audios["max_feats"]  # loss = 0 when min'd
        self.loss_fn *= self.weights


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
            CustomLoss,
            alignment.graph.target_alignments,
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
            CustomLoss,
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


def custom_extract_results(attack):
    results = data.egress.extract.get_unbounded_attack_state(attack)
    results["loss_weightings"] = attack.procedure.tf_run(
        attack.loss[0].c
    )
    return results


def attack_run(master_settings):
    """
    """

    align = master_settings["align"]
    decoder = master_settings["decoder"]
    outdir = master_settings["outdir"]

    outdir = os.path.join(outdir, "unbounded/confidence/weightedmaxmin/")
    outdir = os.path.join(outdir, "{}/".format(align))
    outdir = os.path.join(outdir, "{}/".format(decoder))

    master_settings["outdir"] = outdir

    batch_gen = ALIGNMENT_CHOICES[align](master_settings)

    manager(
        master_settings,
        create_attack_graph,
        batch_gen,
        results_extract_fn=custom_extract_results,
        results_transform_fn=data.egress.transform.unbounded_gen,
    )
    log("Finished run.")


if __name__ == '__main__':

    log("", wrap=True)

    extra_args = {
        'align': [str, "sparse", False, ALIGNMENT_CHOICES.keys()],
    }

    args(attack_run, additional_args=extra_args)



