#!/usr/bin/env python3
import os

# attack def imports
from cleverspeech.graph.AttackConstructors import EvasionAttackConstructor
from cleverspeech.graph import Constraints
from cleverspeech.graph import PerturbationSubGraphs
from cleverspeech.graph import Losses
from cleverspeech.graph import Optimisers
from cleverspeech.graph import Procedures
from cleverspeech.graph import Placeholders


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


LOSSES = {
    "ctc": Losses.CTCLoss,
    "ctc2": Losses.CTCLossV2,
}


def execute(settings, attack_fn, batch_gen):

    # set up the directory we'll use for results

    if not os.path.exists(settings["outdir"]):
        os.makedirs(settings["outdir"], exist_ok=True)

    results_extractor = AttackETLs.convert_evasion_attack_state_to_dict
    results_transformer = AttackETLs.EvasionResults()

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

    attack = EvasionAttackConstructor(sess, batch, feeds)
    attack.add_placeholders(Placeholders.Placeholders)
    attack.add_hard_constraint(
        Constraints.L2,
        r_constant=settings["rescale"],
        update_method=settings["constraint_update"],
    )
    attack.add_perturbation_subgraph(
        PerturbationSubGraphs.Independent
    )
    attack.add_victim(
        Victim.Model,
        decoder=settings["decoder"],
        beam_width=settings["beam_width"]
    )
    attack.add_loss(
        LOSSES[settings["loss"]]
    )
    attack.create_loss_fn()
    attack.add_optimiser(
        Optimisers.AdamIndependentOptimiser,
        learning_rate=settings["learning_rate"]
    )
    attack.add_procedure(
        Procedures.StandardProcedure,
        steps=settings["nsteps"],
        update_step=settings["decode_step"]
    )

    return attack


def attack_run(master_settings):
    """
    CTC Loss attack modified from the original Carlini & Wagner work.
    """

    loss = master_settings["loss"]
    decoder = master_settings["decoder"]
    outdir = master_settings["outdir"]

    outdir = os.path.join(outdir, "evasion/baselines/ctc/")
    outdir = os.path.join(outdir, "{}/".format(loss))
    outdir = os.path.join(outdir, "{}/".format(decoder))

    master_settings["outdir"] = outdir

    batch_gen = batch_generators.standard(master_settings)
    execute(master_settings, create_attack_graph, batch_gen)

    log("Finished run.")


if __name__ == '__main__':

    log("", wrap=True)

    extra_args = {
        "loss": [str, "ctc", False, ["ctc", "ctc2"]],
    }

    args(attack_run, additional_args=extra_args)

