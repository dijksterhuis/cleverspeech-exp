import tensorflow as tf
import numpy as np
import sys
import os
import json

from cleverspeech.Data import Batches
from cleverspeech.Data import Generators
from cleverspeech.Data.Results import SingleJsonDB
from cleverspeech.Models import DeepSpeech
from cleverspeech.Utils import log
from cleverspeech.RuntimeUtils import create_tf_runtime


class OriginalLogitsLoader(object):
    def __init__(self, file_paths, targets, tokens):
        """
        batch singleton.

        :param file_paths: the file paths of the examples in the batch [batch_size]
        :param targets: the target transcriptions [batch size]
        :param tokens: the tokens used in CTC (used in `Targets` class) [1]
        """
        self.size = len(file_paths)
        self.audios = Batches.Audios(file_paths, dtype="int16", quantization=False)
        self.targets = Batches.Targets(targets, tokens)
        self.feeds = Batches.ValidationFeeds(self.audios, self.targets)


def main(indir, batch_size, tokens=" abcdefghijklmnopqrstuvwxyz'-"):

    # Create the factory we'll use to iterate over N examples at a time.

    batch_factory = Generators.BatchGenerator(
        indir,
        None,
        target_phrase="",
        tokens=tokens,
        sort_by_file_size="desc",
    )

    batch_gen = batch_factory.generate(
        OriginalLogitsLoader,
        batch_size=batch_size
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
    indir, batch_size = sys.argv[1:]

    main(indir, int(batch_size))

