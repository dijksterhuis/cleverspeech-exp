#!/usr/bin/env python3
import os

# attack def imports
from cleverspeech.graph.AttackConstructors import UnboundedAttackConstructor
from cleverspeech.graph import Constraints
from cleverspeech.graph import PerturbationSubGraphs
from cleverspeech.graph import Losses
from cleverspeech.graph import Optimisers
from cleverspeech.graph import Procedures
from cleverspeech.graph import Placeholders
from cleverspeech.graph.CTCAlignmentSearch import create_tf_ctc_alignment_search_graph

from cleverspeech.data.ingress.etl import batch_generators
from cleverspeech.data.ingress import Feeds
from cleverspeech.data.egress.Databases import SingleJsonDB
from cleverspeech.data.egress import AttackETLs
from cleverspeech.data.egress.Writers import SingleFileWriter
from cleverspeech.data.egress import Reporting

from cleverspeech.utils.Utils import log
from cleverspeech.utils.runtime.AttackSpawner import AttackSpawner
from cleverspeech.utils.runtime.ExperimentArguments import args

# victim model
from SecEval import VictimAPI as DeepSpeech


ALIGNMENT_CHOICES = {
    "sparse": batch_generators.sparse,
    "ctcalign": batch_generators.standard,
    "dense": batch_generators.dense,
}


def execute(settings, attack_fn, batch_gen):

    # set up the directory we'll use for results

    if not os.path.exists(settings["outdir"]):
        os.makedirs(settings["outdir"], exist_ok=True)

    results_extractor = AttackETLs.convert_unbounded_attack_state_to_dict
    results_transformer = AttackETLs.UnboundedResults()

    file_writer = SingleFileWriter(settings["outdir"], results_transformer)

    # Write the current settings to "settings.json" file.

    settings_db = SingleJsonDB(settings["outdir"])
    settings_db.open("settings").put(settings)
    log("Wrote settings.")

    # Manage GPU memory and CPU processes usage.

    attack_spawner = AttackSpawner(
        gpu_device=settings["gpu_device"],
        max_processes=settings["max_spawns"],
        delay=settings["spawn_delay"],
        file_writer=file_writer,
    )

    with attack_spawner as spawner:

        for b_id, batch in batch_gen:

            log("Running for Batch Number: {}".format(b_id), wrap=True)
            attack_args = (settings, attack_fn, batch, results_extractor)
            spawner.spawn(attack_args)

    # Run the stats function on all successful examples once all attacks
    # are completed.
    Reporting.generate_stats_file(settings["outdir"])


def create_attack_graph(sess, batch, settings):

    feeds = Feeds.Attack(batch)

    attack = UnboundedAttackConstructor(sess, batch, feeds)
    attack.add_placeholders(Placeholders.Placeholders)
    attack.add_perturbation_subgraph(
        PerturbationSubGraphs.Independent
    )
    attack.add_victim(
        DeepSpeech.Model,
        decoder=settings["decoder"],
        beam_width=settings["beam_width"]
    )

    if settings["align"] == "ctcalign":

        alignment = create_tf_ctc_alignment_search_graph(attack, batch)

        attack.add_loss(
            Losses.CWMaxDiff,
            alignment.graph.target_alignments,
            k=settings["kappa"]
        )
        attack.add_optimiser(
            Optimisers.AdamIndependentOptimiser,
            learning_rate=settings["learning_rate"]
        )
        attack.add_procedure(
            Procedures.CTCAlignUnbounded,
            alignment_graph=alignment,
            steps=settings["nsteps"],
            update_step=settings["decode_step"]
        )

    else:

        attack.add_loss(
            Losses.CWMaxDiff,
            attack.placeholders.targets,
            k=settings["kappa"]
        )
        attack.add_optimiser(
            Optimisers.AdamIndependentOptimiser,
            learning_rate=settings["learning_rate"]
        )
        attack.add_procedure(
            Procedures.Unbounded,
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
    decoder = master_settings["decoder"]
    kappa = master_settings["kappa"]
    outdir = master_settings["outdir"]

    outdir = os.path.join(outdir, "unbounded/baselines/cwmaxdiff/")
    outdir = os.path.join(outdir, "{}/".format(align))
    outdir = os.path.join(outdir, "{}/".format(decoder))
    outdir = os.path.join(outdir, "{}/".format(kappa))

    master_settings["outdir"] = outdir

    batch_gen = ALIGNMENT_CHOICES[align](master_settings)

    execute(master_settings, create_attack_graph, batch_gen,)
    log("Finished run.")


if __name__ == '__main__':

    log("", wrap=True)

    extra_args = {
        'align': [str, "sparse", False, ALIGNMENT_CHOICES.keys()],
        "kappa": [float, 5.0, False, None],
    }

    args(attack_run, additional_args=extra_args)



