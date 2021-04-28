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
OUTDIR = "./adv/baselines/biggiomaxmin/"
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


def create_regular_attack_graph(sess, batch, settings):
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
        tokens=settings["tokens"],
        decoder=settings["decoder_type"],
        beam_width=settings["beam_width"]
    )

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
        Procedures.StandardProcedure,
        steps=settings["nsteps"],
        update_step=settings["decode_step"]
    )

    return attack


def create_ctcalign_attack_graph(sess, batch, settings):
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
        tokens=settings["tokens"],
        decoder=settings["decoder_type"],
        beam_width=settings["beam_width"]
    )

    alignment = create_tf_ctc_alignment_search_graph(attack, batch, feeds)

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
        Procedures.StandardCTCAlignProcedure,
        alignment_graph=alignment,
        steps=settings["nsteps"],
        update_step=settings["decode_step"]
    )

    return attack


def sparse_beam_search_run(master_settings):
    # for run in range(0, N_RUNS * 2 + 1, 2):

    outdir = os.path.join(OUTDIR, "beam/")
    outdir = os.path.join(outdir, "sparse/")

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
        "decoder_type": "batch",
        "max_examples": MAX_EXAMPLES,
        "max_targets": MAX_TARGETS,
        "max_audio_length": MAX_AUDIO_LENGTH,
    }

    settings.update(master_settings)
    batch_gen = batch_generators.sparse(settings)
    execute(settings, create_regular_attack_graph, batch_gen)
    log("Finished run.") # {}.".format(run))


def sparse_greedy_search_run(master_settings):
    # for run in range(0, N_RUNS * 2 + 1, 2):

    outdir = os.path.join(OUTDIR, "greedy/")
    outdir = os.path.join(outdir, "sparse/")

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
        "decoder_type": "greedy",
        "max_examples": MAX_EXAMPLES,
        "max_targets": MAX_TARGETS,
        "max_audio_length": MAX_AUDIO_LENGTH,
    }

    settings.update(master_settings)
    batch_gen = batch_generators.sparse(settings)
    execute(settings, create_regular_attack_graph, batch_gen)
    log("Finished run.") # {}.".format(run))


def dense_beam_search_run(master_settings):

    outdir = os.path.join(OUTDIR, "beam/")
    outdir = os.path.join(outdir, "dense/")

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
        "decoder_type": "batch",
        "max_examples": MAX_EXAMPLES,
        "max_targets": MAX_TARGETS,
        "max_audio_length": MAX_AUDIO_LENGTH,
    }

    settings.update(master_settings)
    batch_gen = batch_generators.dense(settings)
    execute(settings, create_regular_attack_graph, batch_gen)
    log("Finished run.") # {}.".format(run))


def dense_greedy_search_run(master_settings):

    outdir = os.path.join(OUTDIR, "greedy/")
    outdir = os.path.join(outdir, "dense/")

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
        "decoder_type": "greedy",
        "max_examples": MAX_EXAMPLES,
        "max_targets": MAX_TARGETS,
        "max_audio_length": MAX_AUDIO_LENGTH,
    }

    settings.update(master_settings)
    batch_gen = batch_generators.dense(settings)
    execute(settings, create_regular_attack_graph, batch_gen)
    log("Finished run.") # {}.".format(run))


def ctcalign_beam_search_run(master_settings):
    """
    Use CTC Loss to optimise some target logits for us. This is quick and simple
    but the deocder confidence scores are usually a lot lower than the example's
    original transcription score.

    :return: None
    """

    outdir = os.path.join(OUTDIR, "beam/")
    outdir = os.path.join(outdir, "ctcalign/")

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
        "decoder_type": "batch",
        "max_examples": MAX_EXAMPLES,
        "max_targets": MAX_TARGETS,
        "max_audio_length": MAX_AUDIO_LENGTH,
    }

    settings.update(master_settings)
    batch_gen = batch_generators.standard(settings)
    execute(
        settings,
        create_ctcalign_attack_graph,
        batch_gen,
    )
    log("Finished run.") # {}.".format(run))


def ctcalign_greedy_search_run(master_settings):
    """
    Use CTC Loss to optimise some target logits for us. This is quick and simple
    but the deocder confidence scores are usually a lot lower than the example's
    original transcription score.

    :return: None
    """

    outdir = os.path.join(OUTDIR, "greedy/")
    outdir = os.path.join(outdir, "ctcalign/")

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
        "decoder_type": "greedy",
        "max_examples": MAX_EXAMPLES,
        "max_targets": MAX_TARGETS,
        "max_audio_length": MAX_AUDIO_LENGTH,
    }

    settings.update(master_settings)
    batch_gen = batch_generators.standard(settings)
    execute(
        settings,
        create_ctcalign_attack_graph,
        batch_gen,
    )
    log("Finished run.") # {}.".format(run))


if __name__ == '__main__':

    log("", wrap=True)

    experiments = {
        "ctcalign-beam": ctcalign_beam_search_run,
        "ctcalign-greedy": ctcalign_greedy_search_run,
        "dense-beam": dense_beam_search_run,
        "dense-greedy": dense_greedy_search_run,
        "sparse-beam": sparse_beam_search_run,
        "sparse-greedy": sparse_greedy_search_run,
    }

    args(experiments)


