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
NUMB_STEPS = DECODING_STEP ** 2
BATCH_SIZE = 10

# extreme run settings
LOSS_UPDATE_THRESHOLD = 10.0


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
        Victim.Model,
        tokens=settings["tokens"],
        decoder=settings["decoder_type"],
        beam_width=settings["beam_width"]
    )

    if settings["align"] == "ctcalign":

        alignment = create_tf_ctc_alignment_search_graph(sess, batch)

        attack.add_loss(
            Losses.BiggioMaxMin,
            alignment.graph.target_alignments,
        )
        attack.create_loss_fn()
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
            Losses.BiggioMaxMin,
            attack.placeholders.targets,
        )
        attack.create_loss_fn()
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
    """

    align = master_settings["align"]
    decoder = master_settings["decoder"]

    outdir = os.path.join(OUTDIR, "unbounded/baselines/biggio/")
    outdir = os.path.join(outdir, "{}/".format(align))
    outdir = os.path.join(outdir, "{}/".format(decoder))

    settings = {
        "audio_indir": AUDIOS_INDIR,
        "targets_path": TARGETS_PATH,
        "outdir": outdir,
        "batch_size": BATCH_SIZE,
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
        "max_examples": MAX_EXAMPLES,
        "max_targets": MAX_TARGETS,
        "max_audio_length": MAX_AUDIO_LENGTH,
        "align": align,
        "decoder_type": decoder,
    }

    settings.update(master_settings)

    if align == "ctcalign":
        batch_gen = batch_generators.standard(settings)

    elif align == "sparse":
        batch_gen = batch_generators.sparse(settings)

    elif align == "dense":
        batch_gen = batch_generators.dense(settings)

    else:
        raise NotImplementedError("Incorrect choice for --align argument.")

    execute(settings, create_attack_graph, batch_gen,)
    log("Finished run.")


if __name__ == '__main__':

    log("", wrap=True)

    extra_args = {
        'align': [str, "sparse", False, ["sparse", "ctcalign", "dense"]],
        "decoder": [str, "batch", False, ["greedy", "batch", "ds", "tf"]],
    }

    args(attack_run, additional_args=extra_args)



