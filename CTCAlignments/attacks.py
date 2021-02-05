#!/usr/bin/env python3
import os

from cleverspeech.graph.GraphConstructor import Constructor
from cleverspeech.graph import Constraints
from cleverspeech.graph import Graphs
from cleverspeech.graph import Optimisers
from cleverspeech.graph import Procedures
from cleverspeech.graph import Outputs
from cleverspeech.data import Feeds
from cleverspeech.data import ETL
from cleverspeech.data import Generators
from cleverspeech.utils.Utils import log, args

from SecEval import VictimAPI as DeepSpeech


from boilerplate import execute

# local attack classes
import custom_defs
import custom_etl

GPU_DEVICE = 0
MAX_PROCESSES = 3
SPAWN_DELAY = 60 * 5

TOKENS = " abcdefghijklmnopqrstuvwxyz'-"
BEAM_WIDTH = 500

AUDIOS_INDIR = "./samples/all/"
TARGETS_PATH = "./samples/cv-valid-test.csv"
OUTDIR = "./adv/ctc-alignments/"

# targets search parameters
MAX_EXAMPLES = 1000
MAX_TARGETS = 1000
MAX_AUDIO_LENGTH = 120000

RESCALE = 0.95
CONSTRAINT_UPDATE = "geom"
LEARNING_RATE = 10
NUMB_STEPS = 10000
DECODING_STEP = 10
BATCH_SIZE = 10

# extreme run settings
LOSS_UPDATE_THRESHOLD = 10.0
LOSS_UPDATE_NUMB_STEPS = 50000

N_RUNS = 1


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


def ctc_dense_alignment_run(master_settings):
    """
    Repeating characters (linear expansion) from a target transcription to
    create an alignment doesn't help the adversary. There's a 0.5 drop in
    cumulative log probabilities and a tripling of the required modification
    size (relative to maximum possible size).

    We want to maximise the classifier confidence for our attacks (see Wild
    Patterns -- Biggio et al. about the perturbation size misconception).

    CTC Loss is used to perform maximum likelihood optimisation. CTC
    Loss is usually used with respect to a target transcription (where the
    length of the target transcription is less than or equal to the number of
    frames). But we can trick CTC Loss into optimising an alignment with two key
    changes.

    The below settings are an edge case for CTC Loss use.
    ```
        preprocess_collapse_repeated=False,
        ctc_merge_repeated=True,
    ```

    Enabling them means that:
    (a) duplicated characters in a target sequence are not collapsed during
    pre-processing
    (b) duplicated characters are not merged during optimisation.

    But we can't pass in an alignment sequence as a target without tricking CTC
    Loss into treating the blank token `-` as a real/valid character.

    The network's softmax matrix output is extended with an M (n_frames) length
    vector of zero values. These extra values act as a "dummy" blank token,
    which will never be likely.

    Now we can include the blank `-` token in the target sequence.

    This obviously modifies one of the conditions for CTC Loss -- now the target
    sequence length must be equal to the number of audio frames.

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
            beam_width=settings["beam_width"]
        )

        attack.add_adversarial_loss(custom_defs.RepeatsCTCLoss)

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

    for run in range(0, N_RUNS):

        outdir = os.path.join(OUTDIR, "dense/")
        outdir = os.path.join(outdir, "run_{}/".format(run))

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
        }

        settings.update(master_settings)

        batch_factory = get_dense_batch_factory(settings)

        execute(settings, create_attack_graph, batch_factory)

        log("Finished run {}.".format(run))


def ctc_dense_extreme_alignment_run(master_settings):
    """
    As above, expect only update bounds when loss is below some threshold.
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
            beam_width=settings["beam_width"]
        )

        attack.add_adversarial_loss(custom_defs.RepeatsCTCLoss)

        attack.create_loss_fn()

        attack.add_optimiser(
            Optimisers.AdamOptimiser,
            learning_rate=settings["learning_rate"]
        )

        attack.add_procedure(
            Procedures.UpdateOnLoss,
            steps=settings["nsteps"],
            decode_step=settings["decode_step"],
            loss_lower_bound=settings["loss_threshold"],
        )

        attack.add_outputs(
            Outputs.Base,
            settings["outdir"],
        )

        return attack

    for run in range(0, N_RUNS):

        outdir = os.path.join(OUTDIR, "dense/")
        outdir = os.path.join(outdir, "run_{}/".format(run))

        settings = {
            "audio_indir": AUDIOS_INDIR,
            "targets_path": TARGETS_PATH,
            "outdir": outdir,
            "batch_size": BATCH_SIZE,
            "tokens": TOKENS,
            "nsteps": LOSS_UPDATE_NUMB_STEPS,
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
            "loss_threshold": LOSS_UPDATE_THRESHOLD,
        }

        settings.update(master_settings)

        batch_factory = get_dense_batch_factory(settings)

        execute(settings, create_attack_graph, batch_factory)

        log("Finished run {}.".format(run))


def get_sparse_batch_factory(settings):

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


def ctc_sparse_alignment_run(master_settings):
    """
    A confident sparse alignment can be derived from the argmax of a dummy
    softmax matrix (initially zero valued) by solving the following optimisation
    goal:

        minimise_{Z} CTC Loss(Z, t')

    By optimising a dummy matrix, we end up with a matrix where the most likely
    characters per frame are (usually) the targets we want.

    *Most importantly*, this optimisation is independent of any audio example --
    the alignment we find will be the most likely alignment for _this_ model and
    for _this_ target transcription.

    Again, we want to maximise the classifier confidence for our attacks
    (see Wild Patterns -- Biggio et al. about the perturbation size
    misconception).

    CTC Loss is used to perform maximum likelihood optimisation. CTC
    Loss is usually used with respect to a target transcription (where the
    length of the target transcription is less than or equal to the number of
    frames). But we can trick CTC Loss into optimising an alignment with two key
    changes.

    The below settings are an edge case for CTC Loss use.
    ```
        preprocess_collapse_repeated=False,
        ctc_merge_repeated=True,
    ```

    Enabling them means that:
    (a) duplicated characters in a target sequence are not collapsed during
    pre-processing
    (b) duplicated characters are not merged during optimisation.

    But we can't pass in an alignment sequence as a target without tricking CTC
    Loss into treating the blank token `-` as a real/valid character.

    The network's softmax matrix output is extended with an M (n_frames) length
    vector of zero values. These extra values act as a "dummy" blank token,
    which will never be likely.

    Now we can include the blank `-` token in the target sequence.

    This obviously modifies one of the conditions for CTC Loss -- now the target
    sequence length must be equal to the number of audio frames.

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
            beam_width=settings["beam_width"]
        )

        alignment = Constructor(attack.sess, batch)
        alignment.add_graph(custom_defs.CTCSearchGraph, attack)
        alignment.add_adversarial_loss(custom_defs.AlignmentLoss)
        alignment.create_loss_fn()
        alignment.add_optimiser(custom_defs.CTCAlignmentOptimiser)

        attack.add_adversarial_loss(
            custom_defs.RepeatsCTCLoss,
            alignment=alignment.graph.target_alignments,
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
            decode_step=settings["decode_step"],
        )

        attack.add_outputs(
            Outputs.Base,
            settings["outdir"],
        )

        return attack

    for run in range(0, N_RUNS):

        outdir = os.path.join(OUTDIR, "sparse/")
        outdir = os.path.join(outdir, "run_{}/".format(run))

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
        }

        settings.update(master_settings)

        batch_factory = get_sparse_batch_factory(settings)

        execute(settings, create_attack_graph, batch_factory)

        log("Finished run {}.".format(run))


def ctc_sparse_extreme_alignment_run(master_settings):
    """
    As above, but this time we define `success` when the current loss is below a
    specified threshold.
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
            beam_width=settings["beam_width"]
        )

        alignment = Constructor(attack.sess, batch)
        alignment.add_graph(custom_defs.CTCSearchGraph, attack)
        alignment.add_adversarial_loss(custom_defs.AlignmentLoss)
        alignment.create_loss_fn()
        alignment.add_optimiser(custom_defs.CTCAlignmentOptimiser)

        attack.add_adversarial_loss(
            custom_defs.RepeatsCTCLoss,
            alignment=alignment.graph.target_alignments,
        )

        attack.create_loss_fn()

        attack.add_optimiser(
            Optimisers.AdamOptimiser,
            learning_rate=settings["learning_rate"]
        )

        attack.add_procedure(
            custom_defs.CTCAlignmentsUpdateOnLoss,
            alignment_graph=alignment,
            steps=settings["nsteps"],
            decode_step=settings["decode_step"],
            loss_lower_bound=settings["loss_threshold"],
        )

        attack.add_outputs(
            Outputs.Base,
            settings["outdir"],
        )

        return attack

    for run in range(0, N_RUNS):

        outdir = os.path.join(OUTDIR, "sparse-extreme/")
        outdir = os.path.join(outdir, "run_{}/".format(run))

        settings = {
            "audio_indir": AUDIOS_INDIR,
            "targets_path": TARGETS_PATH,
            "outdir": outdir,
            "batch_size": BATCH_SIZE,
            "tokens": TOKENS,
            "nsteps": LOSS_UPDATE_NUMB_STEPS,
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
            "loss_threshold": LOSS_UPDATE_THRESHOLD,
        }

        settings.update(master_settings)

        batch_factory = get_sparse_batch_factory(settings)

        execute(settings, create_attack_graph, batch_factory)

        log("Finished run {}.".format(run))


if __name__ == '__main__':

    experiments = {
        "dense": ctc_dense_alignment_run,
        "sparse": ctc_sparse_alignment_run,
        "dense-extreme": None,
        "sparse-extreme": ctc_sparse_extreme_alignment_run,
    }

    args(experiments)

