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

