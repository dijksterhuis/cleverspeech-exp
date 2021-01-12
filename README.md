# cleverSpeech Experiments

A bunch of experiments devised with the [cleverSpeech](https://github.com/dijksterhuis/cleverSpeech) 
repo as part of my PhD.

### Installing
Docker images are located here (TODO).

Otherwise, follow the installation instructions for [cleverSpeech](https://github.com/dijksterhuis/cleverSpeech).

### Experiments

All attacks are performed with a hard L2 norm constraint (no soft constraint in the adversarial loss).
Currently working on a few things on the back of the [CTCHiScores](https://github.com/dijksterhuis/cleverSpeechExperiments#ctchiscores)
experiment.

#### Baselines
Simple CTC attack based largely on the work of Nicholas Carlini and David Wagner.
Used as a baseline comparison for all other work (attack success rates, perturbation size etc.).

#### AdditiveSynthesis
Explores the effects of using additive synthesis to generate pertubations.
Additive synthesis doesn't usually help the adversary, but some attacks do seem to work.

This experiment highlighted an issue Lea Schoenheer discussed in a recent paper
(Adam optimiser struggles to optimise batches of variable length sequences).
A potential work around could implement Adam variables (epsilon etc.) in a batch-wise manner.

#### CTCHiScores
Find a high confidence alignment (in terms of the Decoder log prob. score) and optimise with CTC-Loss.
Finds solutions much faster than existing alignment based attacks without multiple stages.

Dense alignments (e.g. ooooppppeeeennnn) have higher decoder confidence scores than sparse alignments (e.g. o---p---e---n---).
This could have implications for CTC as there are a lot of alignments CTC doesn't need to consider during optimisation 
(dense vs. sparse).

#### SimpleHiScores
Similar to CTCHiScores, but directly optimising the output logits vs. high confidence target logits.
Doesn't work very well as the optimisation doesn't seem to converge 
(irrelevant classes per time-step are being modified more often than not, so the attack gets stuck).

