# cleverSpeech Experiments

A bunch of experiments devised with the [cleverSpeech](https://github.com/dijksterhuis/cleverSpeech) 
repo as part of my PhD.

## Experiments

Usually use a hard L2 norm constraint. Some experiments look at regularisation losses, but they're
generally less interesting. Currently working on a few things on the back of the
[CTCHiScores](https://github.com/dijksterhuis/cleverSpeechExperiments#ctchiscores) experiment.

### AlignmentTargeting
Various search methods to find high confidence alignments (i.e. high decoder log prob. scores).

### Baselines
Simple CTC attack based largely on the work of Nicholas Carlini and David Wagner. Used as a baseline
comparison for all other work (attack success rates, perturbation size etc.).

### Synthesis
Explores the effects of using various synthesis methods (Additive, Spectral) to generate
perturbations. Additive synthesis doesn't usually help the adversary, but some attacks do seem to
work. Spectral does a better job, potentially because the adversary is optimising in 2D space rather
than 1D.

This additive experiments highlighted an issue Lea Schoenherr discussed in a recent paper (Adam
optimiser struggles to optimise batches of variable length sequences). A potential work around could
implement Adam variables (epsilon etc.) in a batch-wise manner.

### CTCHiScores
Find a high confidence alignment (in terms of the Decoder log prob. score) and optimise with
CTC-Loss. Finds solutions much faster than existing alignment based attacks without multiple stages.

Dense alignments (e.g. ooooppppeeeennnn) have higher decoder confidence scores than sparse
alignments (e.g. o---p---e---n---). This could have implications for CTC as there are a lot of
alignments CTC doesn't need to consider during optimisation (dense vs. sparse).

### SimpleHiScores
Similar to CTCHiScores, but directly optimising the output logits vs. high confidence target logits.
Doesn't work very well as the optimisation doesn't seem to converge (irrelevant classes per
time-step are being modified more often than not, so the attack gets stuck).


## Run the code

*N.B.*: I'm in the middle of a big refactor, so these docker instructions are out of date.

Docker images are available [here](https://hub.docker.com/u/dijksterhuis/cleverspeech).

The `latest` or `experiment` tags include the experiments I've run for my PhD work as part of the
[cleverSpeechExperiments](https://github.com/dijksterhuis/cleverSpeechExperiments) repo.
The `build` tag is the basic image with _only_ the
[cleverSpeech](https://github.com/dijksterhuis/cleverSpeech) repo included.

My work is packaged with docker so that:
1. You don't have to go through the same dependency hell I went through.
2. You don't have to worry about getting the right data, checkpoints, commits etc.
3. You can validate my results with the exact set-up I had by running one/two commands.

To start running some experiments with docker:

1. Install the latest version of [docker][10] (at least version `19.03`).
2. Install and configure the [NVIDIA container runtime][8].
3. Run the container (the image itself will be pulled automatically):
```bash
docker run \
    -it \
    --rm \
    --gpus all \
    -e LOCAL_UID=$(id -u ${USER}) \
    -e LOCAL_GID=$(id -g ${USER}) \
    -v path/to/original/samples/dir:/home/cleverspeech/cleverSpeech/samples:ro \
    -v path/to/output/dir:/home/cleverspeech/cleverSpeech/adv:rw \
    dijksterhuis/cleverspeech:latest
```
4. Run one of the scripts from [cleverSpeechExperiments](https://github.com/dijksterhuis/cleverSpeechExperiments)
```bash
python3 ./experiments/Baselines/attacks.py baseline
```

The `LOCAL_UID` and `LOCAL_GID` environment variables must be set. They're used to map file
permissions in `/home/cleverspeech` user to your current user, otherwise you have to mess around
with root file permission problems on any generated data.

Check out the `attacks.py` scripts for additional usage, especially pay attention to the `settings`
dictionaries, any `GLOBAL_VARS` (top of the scripts) and the `boilerplate.py` files. Feel free to
email me with any queries.

### Notes / Gotchas

**1**: Only 16 bit signed integer audio files are supported -- i.e. mozilla common voice v1.

**2**: Integrity of the adversarial examples is an ongoing issue when using the
`deepspeech`/`deepspeech-gpu` python library (the one installed with `pip`). The DeepSpeech source
ingests `tf.float32` inputs `-2^15 <= x <= 2^15 -1`, but the `deepspeech` library _only_ ingests 16
bit integers. Use the [`classify.py` script](cleverspeech/Evaluation/classify.py) in
`./cleverspeech/Evaluation/` to validate outputs.

**3**: I run my experiments in sets (not batches!) of 10 examples. Adam struggles to optimise
as it's built for batch-wise learning rate tuning, but each of our examples are independent members
of a set (Lea Schoenherr's [recent paper][12] talks about this briefly).

**4**: `.jenkins` contains all the build and execution pipeline steps for the docker images and
experiments.

### Non-Docker Usage

You need to have both the [cleverSpeech](https://github.com/dijksterhuis/cleverSpeech) and
[DeepSpeechAdversary](https://github.com/dijksterhuis/DeepSpeechAdversary) repos cloned on your
system.

Install the necessary requirements for each repo. Then add both to your `PYTHONPATH`.
Make sure you've set the `DEEPSPEECH_CHECKPOINT_DIR` and `DEEPSPEECH_MODEL_DIR` environment variables
(see [here](https://github.com/dijksterhuis/DeepSpeechAdversary/blob/adversarial-v0.4.1/DeepSpeechSecEval/VictimAPI.py#L127)).

After that, everything should _just work_.


### Citations / Licenses / Sourced Works

TODO: Update licenses.

I've modified the following works, many thanks to the authors:
- [Carlini & Wagner][0]
- [magneta/ddsp][4]


[0]: https://arxiv.org/abs/1801.01944
[2]: https://arxiv.org/abs/1608.04644
[3]: https://arxiv.org/abs/1712.03141
[4]: https://github.com/magenta/ddsp
[5]: https://arxiv.org/abs/1902.06705
[6]: https://hub.docker.com/r/dijksterhuis/cleverspeech
[7]: https://github.com/dijksterhuis/cleverSpeech/packages
[8]: https://github.com/NVIDIA/nvidia-container-runtime
[9]: https://whoami.dijksterhuis.co.uk
[10]: https://docker.com
[11]: https://github.com/dijksterhuis/cleverSpeech/packages/336838
[12]: https://arxiv.org/abs/2005.14611
