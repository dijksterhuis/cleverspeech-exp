import tensorflow as tf
import sys
import os

from cleverspeech.data import ETL
from cleverspeech.data import Generators
from cleverspeech.data import Feeds
from cleverspeech.data.Results import SingleJsonDB
from SecEval import VictimAPI as DeepSpeech
from cleverspeech.utils.Utils import log
from cleverspeech.utils.RuntimeUtils import create_tf_runtime


def main(batch_size=1, tokens=" abcdefghijklmnopqrstuvwxyz'-"):

    # Create the factory we'll use to iterate over N examples at a time.

    audio_etl = ETL.AllAudioFilePaths(
        "./tmp-analysis/",
        4000,
        filter_term=".wav",
        max_samples=120000
    )

    all_audio_file_paths = audio_etl.extract().transform().load()

    targets_etl = ETL.AllTargetPhrases(
        "./samples/cv-valid-test.csv", 1000,
    )
    all_targets = targets_etl.extract().transform().load()

    # Generate the batches in turn, rather than all in one go ...

    batch_factory = Generators.BatchGenerator(
        all_audio_file_paths, all_targets, batch_size
    )

    # ... To save resources by only running the final ETLs on a batch of data

    batch_gen = batch_factory.generate(
        ETL.AudioExamples, ETL.TargetPhrases, Feeds.Validation
    )

    for b_id, batch in batch_gen:

        tf_session, tf_device = create_tf_runtime()
        with tf_session as sess, tf_device:

            ph_examples = tf.placeholder(tf.float32, shape=[batch.size, batch.audios.max_length])
            ph_lens = tf.placeholder(tf.float32, shape=[batch.size])

            model = DeepSpeech.Model(sess, ph_examples, batch, tokens=tokens, beam_width=500)

            batch.feeds.create_feeds(ph_examples, ph_lens)

            decodings, probs = model.inference(
                batch,
                feed=batch.feeds.examples,
                decoder="batch",
                top_five=False
            )

            raw, smax = sess.run(
                [
                    tf.transpose(model.raw_logits, [1, 0, 2]),
                    model.logits
                ],
                feed_dict=batch.feeds.examples
            )

            outdir = "original-logits/"

            if not os.path.exists(outdir):
                os.mkdir(outdir)

            for idx, basename in enumerate(batch.audios.basenames):

                stats = {
                    "basename": basename,
                    "decoding": decodings[idx],
                    "score": probs[idx],
                    "raw_logits": raw[idx],
                    "softmax": smax[idx],
                    "size": batch.audios.actual_lengths[idx],
                    "n_feats": batch.audios.alignment_lengths[idx] + 1,
                }

                example_db = SingleJsonDB(outdir)
                example_db.open(basename.rstrip(".wav")).put(stats)

                log("Processed file: {b}".format(b=basename), wrap=False)


if __name__ == '__main__':
    #indir, batch_size = sys.argv[1:]

    main()

