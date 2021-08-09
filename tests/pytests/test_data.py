#!/usr/bin/env python3
from cleverspeech import data
from cleverspeech.utils.runtime.ExperimentArguments import args
from cleverspeech.utils.Utils import log


def manager(batch_gen):

    for b_id, batch in batch_gen:
        log("loaded batch id: {}".format(b_id))

# ==============================================================================


DATA_CHOICES = {
    "mcv_v1": data.ingress.mcv_v1.BatchIterator,
}


def attack_run(master_settings):

    batch_gen = DATA_CHOICES[master_settings["etl"]](master_settings)
    manager(batch_gen)


if __name__ == '__main__':

    log("", wrap=True)

    extra_args = {
        "etl": [str, "mcv_v1", False, DATA_CHOICES.keys()],
    }

    args(attack_run, additional_args=extra_args)

