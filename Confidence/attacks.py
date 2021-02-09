#!/usr/bin/env python3
import os

from cleverspeech.data import Generators
from cleverspeech.data import Feeds
from cleverspeech.graph.GraphConstructor import Constructor
from cleverspeech.graph import Constraints
from cleverspeech.graph import Graphs
from cleverspeech.graph import Losses
from cleverspeech.graph import Optimisers
from cleverspeech.graph import Procedures
from cleverspeech.graph import Outputs

from SecEval import VictimAPI as DeepSpeech
from cleverspeech.data import ETL
from cleverspeech.utils.Utils import log, args

from boilerplate import execute

# custom attack classes and data handling
import custom_defs
import custom_etl


GPU_DEVICE = 0
MAX_PROCESSES = 3
SPAWN_DELAY = 30

TOKENS = " abcdefghijklmnopqrstuvwxyz'-"
BEAM_WIDTH = 500

INDIR = "./target-logits/"
OUTDIR = "./adv/confidence/"

# targets search parameters
MAX_EXAMPLES = 1000

RESCALE = 0.95
CONSTRAINT_UPDATE = "geom"
LEARNING_RATE = 10
NUMB_STEPS = 20000
DECODING_STEP = 10
BATCH_SIZE = 10
LOGITS_SEARCH = "sbgs"

N_RUNS = 1


AUDIOS_INDIR = "./samples/all/"
TARGETS_PATH = "./samples/cv-valid-test.csv"
MAX_TARGETS = 1000
MAX_AUDIO_LENGTH = 120000


def get_sbgs_batch_factory(settings):
    # get a N samples of all the data

    target_logits_etl = custom_etl.AllTargetLogitsAndAudioFilePaths(
        settings["indir"], MAX_EXAMPLES, filter_term="latest"
    )
    all_data = target_logits_etl.extract().transform().load()

    # Generate the batches in turn, rather than all in one go ...
    # ... To save resources by only running the final ETLs on a batch of data

    batch_factory = custom_etl.LogitsBatchGenerator(
        all_data, settings["batch_size"]
    )

    log(
        "New Run",
        "Number of test examples: {}".format(batch_factory.numb_examples),
        ''.join(["{k}: {v}\n".format(k=k, v=v) for k, v in settings.items()]),
    )
    return batch_factory


def squared_diff_loss(master_settings):
    """
    Non functioning.

    :return: None
    """
    def create_attack_graph(sess, batch, settings):
        attack = Constructor(sess, batch)

        attack.add_hard_constraint(
            Constraints.L2,
            r_constant=settings["rescale"],
            update_method=settings["constraint_update"]
        )

        attack.add_graph(
            custom_defs.HiScoresAttack
        )

        attack.add_victim(
            DeepSpeech.Model,
            tokens=settings["tokens"],
            beam_width=settings["beam_width"]
        )

        attack.add_adversarial_loss(
            custom_defs.HiScoresAbsLoss,
        )
        attack.create_loss_fn()

        attack.add_optimiser(
            Optimisers.AdamOptimiser,
            learning_rate=settings["learning_rate"],
        )

        attack.add_procedure(
            Procedures.UpdateOnDecoding,
            steps=settings["nsteps"],
            decode_step=settings["decode_step"]
        )

        attack.add_outputs(
            Outputs.Base,
            settings["outdir"],
        )

        return attack

    for run in range(0, N_RUNS):

        outdir = os.path.join(OUTDIR, "squared_diff/")
        outdir = os.path.join(outdir, "{}/".format(LOGITS_SEARCH))
        outdir = os.path.join(outdir, "run_{}/".format(run))

        settings = {
            "indir": os.path.join(INDIR, "{}/".format(LOGITS_SEARCH)),
            "outdir": outdir,
            "batch_size": BATCH_SIZE,
            "tokens": TOKENS,
            "nsteps": NUMB_STEPS,
            "decode_step": DECODING_STEP,
            "beam_width": BEAM_WIDTH,
            "rescale": RESCALE,
            "constraint_update": "geom",
            "learning_rate": LEARNING_RATE,
            "logits_search": LOGITS_SEARCH,
            "gpu_device": GPU_DEVICE,
            "max_spawns": MAX_PROCESSES,
            "spawn_delay": SPAWN_DELAY,
        }
        settings.update(master_settings)

        batch_factory = get_sbgs_batch_factory(settings)
        batch_gen = batch_factory.generate(
            ETL.AudioExamples, custom_etl.TargetLogits, custom_defs.HiScoresFeed
        )

        execute(settings, create_attack_graph, batch_gen)

        log("Finished run {}.".format(run))


def get_normal_batch_generator(settings):

    # get N samples of all the data. alsp make sure to limit example length,
    # otherwise we'd have to do adaptive batch sizes.

    audio_etl = ETL.AllAudioFilePaths(
        settings["audio_indir"],
        settings["max_examples"],
        filter_term=".wav",
        max_samples=settings["max_audio_length"]
    )

    all_audio_file_paths = audio_etl.extract().transform().load()

    targets_etl = ETL.AllTargetPhrases(
        settings["targets_path"], settings["max_targets"],
    )
    all_targets = targets_etl.extract().transform().load()

    # Generate the batches in turn, rather than all in one go ...

    batch_factory = Generators.BatchGenerator(
        all_audio_file_paths, all_targets, settings["batch_size"]
    )

    # ... To save resources by only running the final ETLs on a batch of data

    batch_gen = batch_factory.generate(
        ETL.AudioExamples, ETL.TargetPhrases, Feeds.Attack
    )

    log(
        "New Run",
        "Number of test examples: {}".format(batch_factory.numb_examples),
        ''.join(["{k}: {v}\n".format(k=k, v=v) for k, v in settings.items()]),
    )
    return batch_gen


def adaptive_kappa_ctc_sparse_run(master_settings):
    """
    MRMaxDiff (extension of CWMaxDiff) with CTCLoss and sparse alignment.

    Aim: Optimise MRMaxDiff whilst regularising with CTCLoss. It seems that this
    actually does the opposite, using MRMaxDiff as a regulariser for CTCLoss to
    force one the specific alignment.

    :return: None
    """
    def create_attack_graph(sess, batch, settings):
        attack = Constructor(sess, batch)

        attack.add_hard_constraint(
            Constraints.L2,
            r_constant=settings["rescale"],
            update_method=settings["constraint_update"],
        )

        attack.add_graph(
            Graphs.SimpleAttack
        )

        attack.add_victim(
            DeepSpeech.Model,
            tokens=settings["tokens"],
            decoder=settings["decoder_type"],
            beam_width=settings["beam_width"]
        )

        alignment = Constructor(attack.sess, batch)
        alignment.add_graph(custom_defs.CTCSearchGraph, attack)
        alignment.add_loss(custom_defs.AlignmentLoss)
        alignment.create_loss_fn()
        alignment.add_optimiser(custom_defs.CTCAlignmentOptimiser)

        attack.add_adversarial_loss(
            custom_defs.AdaptiveKappaCWMaxDiff,
            alignment.graph.target_alignments,
            k=settings["kappa"],
        )
        attack.add_distance_loss(
            Losses.CTCLoss,
            loss_weight=1.0,
        )
        attack.create_loss_fn()
        attack.add_optimiser(
            Optimisers.AdamOptimiser,
            learning_rate=settings["learning_rate"]
        )
        attack.add_procedure(
            custom_defs.CTCAlignmentsUpdateOnDecode,
            alignment_graph=alignment,
            steps=settings["nsteps"],
            decode_step=settings["decode_step"]
        )
        attack.add_outputs(
            Outputs.Base,
            settings["outdir"],
        )

        return attack

    kappa = 0.1

    outdir = os.path.join(OUTDIR, "adaptivekappa-ctcmaxdiff-ctc-sparse/")
    outdir = os.path.join(outdir, "kappa_{}/".format(kappa))

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
        "kappa": float(kappa),
        "decoder_type": "batch",
        "max_examples": MAX_EXAMPLES,
        "max_targets": MAX_TARGETS,
        "max_audio_length": MAX_AUDIO_LENGTH,
    }

    settings.update(master_settings)
    batch_gen = get_normal_batch_generator(settings)

    execute(settings, create_attack_graph, batch_gen)

    log("Finished run {}.".format(kappa))


def adaptive_kappa_sparse_run(master_settings):
    """
    MRMaxDiff (extension of CWMaxDiff) with with a sparse alignment.
    :return: None
    """
    def create_attack_graph(sess, batch, settings):
        attack = Constructor(sess, batch)

        attack.add_hard_constraint(
            Constraints.L2,
            r_constant=settings["rescale"],
            update_method=settings["constraint_update"],
        )

        attack.add_graph(
            Graphs.SimpleAttack
        )

        attack.add_victim(
            DeepSpeech.Model,
            tokens=settings["tokens"],
            decoder=settings["decoder_type"],
            beam_width=settings["beam_width"]
        )

        alignment = Constructor(attack.sess, batch)
        alignment.add_graph(custom_defs.CTCSearchGraph, attack)
        alignment.add_adversarial_loss(custom_defs.AlignmentLoss)
        alignment.create_loss_fn()
        alignment.add_optimiser(custom_defs.CTCAlignmentOptimiser)

        attack.add_adversarial_loss(
            custom_defs.AdaptiveKappaCWMaxDiff,
            alignment.graph.target_alignments,
            k=settings["kappa"]
        )
        attack.create_loss_fn()
        attack.add_optimiser(
            Optimisers.AdamOptimiser,
            learning_rate=settings["learning_rate"]
        )
        attack.add_procedure(
            custom_defs.CTCAlignmentsUpdateOnDecode,
            alignment_graph=alignment,
            steps=settings["nsteps"],
            decode_step=settings["decode_step"]
        )
        attack.add_outputs(
            Outputs.Base,
            settings["outdir"],
        )

        return attack

    kappa = 0.1

    outdir = os.path.join(OUTDIR, "adaptivekappa-ctcmaxdiff-sparse/")
    outdir = os.path.join(outdir, "kappa_{}/".format(kappa))

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
        "kappa": float(kappa),
        "decoder_type": "batch",
        "max_examples": MAX_EXAMPLES,
        "max_targets": MAX_TARGETS,
        "max_audio_length": MAX_AUDIO_LENGTH,
    }

    settings.update(master_settings)
    batch_gen = get_normal_batch_generator(settings)

    execute(settings, create_attack_graph, batch_gen)

    log("Finished run {}.".format(kappa))


def get_dense_batch_factory(settings):

    # get N samples of all the data. alsp make sure to limit example length,
    # otherwise we'd have to do adaptive batch sizes.

    audio_etl = ETL.AllAudioFilePaths(
        settings["audio_indir"],
        settings["max_examples"],
        filter_term=".wav",
        max_samples=settings["max_audio_length"]
    )

    all_audio_file_paths = audio_etl.extract().transform().load()

    targets_etl = ETL.AllTargetPhrases(
        settings["targets_path"], settings["max_targets"],
    )
    all_targets = targets_etl.extract().transform().load()

    # Generate the batches in turn, rather than all in one go ...

    batch_factory = custom_etl.CTCHiScoresBatchGenerator(
        all_audio_file_paths, all_targets, settings["batch_size"]
    )

    # ... To save resources by only running the final ETLs on a batch of data

    batch_gen = batch_factory.generate(
        ETL.AudioExamples, custom_etl.RepeatsTargetPhrases, Feeds.Attack
    )

    log(
        "New Run",
        "Number of test examples: {}".format(batch_factory.numb_examples),
        ''.join(["{k}: {v}\n".format(k=k, v=v) for k, v in settings.items()]),
    )
    return batch_gen


def adaptive_kappa_ctc_dense_run(master_settings):
    """
    Currently broken.
    """
    def create_attack_graph(sess, batch, settings):
        attack = Constructor(sess, batch)

        attack.add_hard_constraint(
            Constraints.L2,
            r_constant=settings["rescale"],
            update_method=settings["constraint_update"],
        )

        attack.add_graph(
            Graphs.SimpleAttack
        )

        attack.add_victim(
            DeepSpeech.Model,
            tokens=settings["tokens"],
            decoder=settings["decoder_type"],
            beam_width=settings["beam_width"]
        )

        attack.add_adversarial_loss(
            custom_defs.AdaptiveKappaCWMaxDiff,
            attack.graph.placeholders.targets,
            k=settings["kappa"]
        )
        attack.add_distance_loss(
            Losses.CTCLoss,
            loss_weight=1.0,
        )
        attack.create_loss_fn()
        attack.add_optimiser(
            Optimisers.AdamOptimiser,
            learning_rate=settings["learning_rate"]
        )
        attack.add_procedure(
            Procedures.UpdateOnDecoding,
            steps=settings["nsteps"],
            decode_step=settings["decode_step"]
        )
        attack.add_outputs(
            Outputs.Base,
            settings["outdir"],
        )

        return attack

    kappa = 0.1

    outdir = os.path.join(OUTDIR, "adaptivekappa-ctcmaxdiff-ctc-dense/")
    outdir = os.path.join(outdir, "kappa_{}/".format(kappa))

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
        "kappa": float(kappa),
        "decoder_type": "batch",
        "max_examples": MAX_EXAMPLES,
        "max_targets": MAX_TARGETS,
        "max_audio_length": MAX_AUDIO_LENGTH,
    }

    settings.update(master_settings)
    batch_gen = get_dense_batch_factory(settings)

    execute(settings, create_attack_graph, batch_gen)

    log("Finished run {}.".format(kappa))


def adaptive_kappa_dense_run(master_settings):
    """
    MRMaxDiff (extension of CWMaxDiff) with a dense alignment.
    """
    def create_attack_graph(sess, batch, settings):
        attack = Constructor(sess, batch)

        attack.add_hard_constraint(
            Constraints.L2,
            r_constant=settings["rescale"],
            update_method=settings["constraint_update"],
        )

        attack.add_graph(
            Graphs.SimpleAttack
        )

        attack.add_victim(
            DeepSpeech.Model,
            tokens=settings["tokens"],
            decoder=settings["decoder_type"],
            beam_width=settings["beam_width"]
        )

        attack.add_adversarial_loss(
            custom_defs.AdaptiveKappaCWMaxDiff,
            attack.graph.placeholders.targets,
            k=settings["kappa"]
        )
        attack.create_loss_fn()
        attack.add_optimiser(
            Optimisers.AdamOptimiser,
            learning_rate=settings["learning_rate"]
        )
        attack.add_procedure(
            Procedures.UpdateOnDecoding,
            steps=settings["nsteps"],
            decode_step=settings["decode_step"]
        )
        attack.add_outputs(
            Outputs.Base,
            settings["outdir"],
        )

        return attack

    kappa = 0.25

    outdir = os.path.join(OUTDIR, "adaptivekappa-ctcmaxdiff-dense/")
    outdir = os.path.join(outdir, "kappa_{}/".format(kappa))

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
        "kappa": float(kappa),
        "decoder_type": "batch",
        "max_examples": MAX_EXAMPLES,
        "max_targets": MAX_TARGETS,
        "max_audio_length": MAX_AUDIO_LENGTH,
    }

    settings.update(master_settings)
    batch_gen = get_dense_batch_factory(settings)

    execute(settings, create_attack_graph, batch_gen)

    log("Finished run {}.".format(kappa))


if __name__ == '__main__':

    experiments = {
        "squared_diff_loss": squared_diff_loss,
        "sparse-adaptivekappa-ctc-maxdiff": adaptive_kappa_ctc_sparse_run,
        "dense-adaptivekappa-ctc-maxdiff": adaptive_kappa_ctc_dense_run,
        "sparse-adaptivekappa-maxdiff": adaptive_kappa_sparse_run,
        "dense-adaptivekappa-maxdiff": adaptive_kappa_dense_run,
    }

    args(experiments)

