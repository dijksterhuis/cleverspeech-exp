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
from SecEval import VictimAPI as Victim

GPU_DEVICE = 0
MAX_PROCESSES = 1
SPAWN_DELAY = 30

AUDIOS_INDIR = "./samples/all/"
TARGETS_PATH = "./samples/cv-valid-test.csv"
OUTDIR = "./adv/"
MAX_EXAMPLES = 100
MAX_TARGETS = 1000
MAX_AUDIO_LENGTH = 120000

TOKENS = " abcdefghijklmnopqrstuvwxyz'-"
BEAM_WIDTH = 500
LEARNING_RATE = 10
CONSTRAINT_UPDATE = "geom"
RESCALE = 0.95
DECODING_STEP = 100
NUMB_STEPS = 5000


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

    if settings["graph_type"] == "batch":
        perturbation_sub_graph_cls = PerturbationSubGraphs.Independent
        optimiser_cls = Optimisers.AdamIndependentOptimiser

    elif settings["graph_type"] == "indy":
        perturbation_sub_graph_cls = PerturbationSubGraphs.Independent
        optimiser_cls = Optimisers.AdamIndependentOptimiser

    else:
        raise NotImplementedError

    feeds = Feeds.Attack(batch)

    attack = EvasionAttackConstructor(sess, batch, feeds)
    attack.add_placeholders(Placeholders.Placeholders)
    attack.add_hard_constraint(
        Constraints.L2,
        r_constant=settings["rescale"],
        update_method=settings["constraint_update"],
    )
    attack.add_perturbation_subgraph(
        perturbation_sub_graph_cls
    )
    attack.add_victim(
        Victim.Model,
        tokens=settings["tokens"],
        decoder=settings["decoder_type"],
        beam_width=settings["beam_width"]
    )
    attack.add_loss(
        Losses.CTCLoss
    )
    attack.create_loss_fn()
    attack.add_optimiser(
        optimiser_cls,
        learning_rate=settings["learning_rate"]
    )
    attack.add_procedure(
        Procedures.StandardProcedure,
        steps=settings["nsteps"],
        update_step=settings["decode_step"]
    )

    return attack


def attack_run(master_settings):

    graph_type = master_settings["graph_type"]
    decoder = master_settings["decoder"]
    nbatch_max = master_settings["nbatch_max"]
    nbatch_step = master_settings["nbatch_step"]

    assert nbatch_max >= 1
    assert nbatch_step >= 1
    assert nbatch_max >= nbatch_step

    for batch_size in range(0, nbatch_max + 1, nbatch_step):

        if batch_size == 0:
            batch_size = 1

        outdir = os.path.join(OUTDIR, "evasion/batch-vs-indy/")
        outdir = os.path.join(outdir, "{}/".format(graph_type))
        outdir = os.path.join(outdir, "{}/".format(decoder))
        outdir = os.path.join(outdir, "{}/".format(batch_size))

        settings = {
            "audio_indir": AUDIOS_INDIR,
            "targets_path": TARGETS_PATH,
            "outdir": outdir,
            "batch_size": batch_size,
            "tokens": TOKENS,
            "nsteps": NUMB_STEPS,
            "decode_step": DECODING_STEP,
            "beam_width": BEAM_WIDTH,
            "constraint_update": CONSTRAINT_UPDATE,
            "rescale": RESCALE,
            "learning_rate": LEARNING_RATE,
            "gpu_device": GPU_DEVICE,
            "max_spawns": MAX_PROCESSES,
            "spawn_delay": SPAWN_DELAY,
            "max_examples": batch_size,
            "max_targets": MAX_TARGETS,
            "max_audio_length": MAX_AUDIO_LENGTH,
            "graph_type": graph_type,
            "decoder_type": decoder,
        }

        settings.update(master_settings)
        batch_gen = batch_generators.standard(settings)
        execute(settings, create_attack_graph, batch_gen)

        log("Finished batch run {}.".format(batch_size))

    log("Finished all runs.")


if __name__ == '__main__':

    log("", wrap=True)

    extra_args = {
        'graph_type': [str, "batch", False, ["batch", "indy"]],
        'nbatch_max': [int, 20, False, None],
        'nbatch_step': [int, 5, False, None],
        'decoder': [str, "batch", False, ["greedy", "batch", "ds", "tf"]],
    }

    args(attack_run, additional_args=extra_args)



