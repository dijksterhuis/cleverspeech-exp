# cleverSpeech Experiments

A bunch of experiments devised with the [cleverSpeech](https://github.com/dijksterhuis/cleverSpeech) 
repo as part of my PhD.

Two scripts (attacks) defined for each directory:
- `attacks.py` contain actual evasion attacks that iteratively clip perturbations under a L2-norm (i.e. projected gradient descent).
- `unbounded.py` do not clip perturbation size, they just keep optimising until the loss is minimised (useful for testing how losses perform independently of evasion).

Generally speaking each script contains the following components:
- a `create_attack_graph` function which is used to define how the atttack should be constructed according to the cleverSpeech api
- an `attack_run` function which basically sets up the paths for results and grabs specific settings like input data generators
- an `extra_args` variable defining extra command line arguments that can be passed to the script (e.g. choosing loss functions)

For a full list of default command line arguments, see the `ExperimentArguments` [module in the cleverSpeech API](https://github.com/dijksterhuis/cleverspeech-py/blob/master/utils/runtime/ExperimentArguments.py).

If an attack requires some custom code that isn't used by other attacks (e.g. `conf-ctcedgecases`) then it's either in the actual `attacks.py`/`unbounded.py` scripts themselves or lives in a `custom_defs.py` file in the attack directory.

Each attack directory also has a `run.groovy` file defining a [Jenkins](https://jenkins.io/) declarative pipeline which I use to orchestrate all my experiment runs. This is useful to find out how to run each attack and/or how to run the docker images to run your own experiments. These are quite cumbersome and hopefully I can change to something like kubeflow or determined.ai type stuff in the future (I should be using a shared library in all honestly but have been making do with PyCharm's Find & Replace over the entire repo for now).

The [testing](./testing) directory is a bunch of Jenkins pipelines that basically run minimal versions of all the attacks to verify commited changes haven't broken anything (they're basically very minimal/hacky integration tests).

----

### Baselines
I use these to compare results and/or to perform experiments looking at non-loss function areas of interest, e.g. seeing what happens when we choose different paths for each loss function.

#### baseline-biggiomaxmin
The Biggio et al. maxmin of class probabilities loss function from Wild Patterns modified for CTC Speech recognition
```
minimize_\delta \sum_{m=1}^{M} - y_{mk'} + max_{k \neq k'} y_{mk}
```
Where `k` is the target class for a particular frame `m`.

Similarly, using the activations/logits:
```
minimize_\delta \sum_{m=1}^{M} - a_{mk'} + max_{k \neq k'} a_{mk}
```
This loss function is unable to find success because optimisation can get stuck on a for particular frame,
making that frame's single class more likely (or making the maximum other class less likely)
without modifying any of the other frames.


#### baseline-ctc
Legit just a modification of the Carlini & Wagner CTC-Loss attack.
```
minimize_\delta CTC-Loss(Y, t')
```


#### baseline-cwmaxdiff
An implementation of the Carlini & Wagner "Improved Loss Function" which is guranated against greedy seach decoding.
This is basically the Biggio MaxMin loss function with the difference between classes capped at `kappa`.
```
minimize_\delta \sum_{m=1}^{M} max(- y_{mk'} + max_{k \neq k'} y_{mk}, -kappa)
```
Where `kappa=0` is good at finding small perturbations and increasing the value of `kappa` makes the difference between classess bigger and bigger.

Similarly, using the activations/logits:
```
minimize_\delta \sum_{m=1}^{M} max(- a_{mk'} + max_{k \neq k'} a_{mk}, -kappa)
```

This does really well for greedy search (when `kappa=0`) but fails against a beam search decoder.
Also, as the attack starts to fail as we increase `kappa` becuase the Biggio MaxMin optimisation issues come into play.

----

### Confidence Attacks
Working towards proper maximum confidence attacks for CTC speech-to-text models (both greedy and beam search decoders).

#### conf-adaptivekappa
Basically attempting to set `kappa` dynamically based on the difference between the `max` and `min` of softmaxes/logitses.
Currently only tested for logitses.
```
minimize_\delta \sum_{m=1}^{M} max(- y_{mk'} + max_{k \neq k'} y_{mk}, -kappa_vect * kappa)
whre kappa_vect = max_{k \neq k'} y_{mk} - min_{k \neq k'} y_{mk}
```

The target class does become `kappa` greater than all other classes, but the distribution (diff between `max` and `min`) of other classes gets completely _flattened_...
For a logits based attack this means the target logit might only be `0.2` greater than other classes becuase the difference between other classes becomes `0.2` itself!

It's a bit weird and I'm not sure how useful this is tbh.

#### conf-biggiomaxofmaxmin
Modification of the Biggio MaxMin/Carlini & Wagner loss functions, where the value of the loss function is the biggest difference between classes.
```
minimize_\delta \sum_{m=1}^{M} max(- y_{mk'} + max_{k \neq k'} y_{mk})
```

Similarly for acitvations/logits
```
minimize_\delta \sum_{m=1}^{M} max(- a_{mk'} + max_{k \neq k'} a_{mk})
```

This is a true Maximum Confidence attack for greedy search decoding
-- optimisation will continue to make this loss negative as it is forced to make the worst performing frame better and better
(the Carlini & Wagner attack isn't really a maximum confidence attack until you set `kappa~=1`,
where it becomes the Biggio MaxMin loss and stops working).

#### conf-ctcedgecases
TODO: Rename

This basically converts CTC Loss into a loss function that requires the selection of a specific path.
In theory it does the same thing as `conf-sumlogprobs` and `conf-cumulativelogprobs` as CTC loss is tricked into thinking that the transcription is `M` length
(there's only one possible way of aligning an `M` length transcription with `M` length softmax/logitses).
So, instead of CTC loss doing
```
minimize_\delta - \sum_{pi \in PI(t)} \sum_{m=1}^{M} log y_mk'
```
It does
```
minimize_\delta - \sum_{pi} \sum_{m=1}^{M} log y_mk'
```
Which is the same as
```
minimize_\delta - \sum_{m=1}^{M} log y_mk'
```
But it takes longer to do optimisation compared to the sum/cumulative log probs attacks (it still creates the state transitions matrix etc.).


#### conf-cumulativelogprobs
Working towards a maximum confidence attack guaranteed against a beam search decoder.
Only implemented for softmax probabilities as we need to do logarithm stuff (need to be between `0` and `1`).
```
minimize_\delta - \sum_{m=1}^{M} log y_mk'
```
This is mostly the same as the `conf-sumlogprobs` attack (results are exactly the same), except for the fact that this does a cumulative sum over the softmax target classes (sum of log probs just sums them straight up).

This implemetation is a bit trickier than the sum of the log probs one so I prefer to use that as it's easier to modify.

#### conf-invertedctc
Basically negative CTC-Loss. A handy way to do an untargeted attack.

#### conf-logprobsgreedydiff
Working towards a maximum confidence attack against beam search decoders.
This basically takes the sum of the log softmax probs and then compares against the most likely class per frame according to greedy search.
Actually doesn't do as well (in terms of beam search scores and perturbation minimisation) compared to the raw sum of the log probs.

#### conf-maxadvctc-mintruectc
Biggio et al say that it's more reasonable to assume that an attack will attempt to maximise classifier confidence of the target class and minimise the likelihood of all other classes (instead of just focusing on making the perturbation size very small).
This basically does that with CTC-Loss, using the _true transcription_ for each example as _all other classes_.

#### conf-sumlogprobs
Exactly the same as the cumulative log probs attack except this just sums over all the log of the softmax probs (the only difference is in how the sum is calculated).
```
minimize_\delta - \sum_{m=1}^{M} log y_mk'
```
Weirdly, this basically does the same thing as the `conf-targetonly` attack for CTC beam search decoders (i.e. only make the target classes more likely)....
But this actually works against a greedy decoder as well, whereas the `conf-targetonly` attack doesn't...

#### conf-targetonly
What ahppens if we perform an attack against a greedy decoder but ignore the _minimise all other classes_ part of a maximum confidence attack?
Turns out, not very much becuase all the other classes can become more likely as well.

Interestingly the sum of the log probs (and variants) basically do this but can actually find success!

#### conf-weightedmaxmin
Another way of attempting to adaptively trick optimisation into focussing on specific frames, doesn't work particularly well it seems (could be because it requires softmax).
```
minimize_\delta \sum_{m=1}^{M} (1-y_{mk'}) . (- y_{mk'} + max_{k \neq k'} y_{mk})
```
----
### Misc
Miscillaneaous attacks that are just used to test stuff.

#### misc-batch-vs-indy
Validating that the entire tensorflow graph is completely independent for all examples.

#### misc-max-gpu-batch-size
Run a quick test to find the maximum number of examples that can be processed on one GPU until OOM errors start happening.




