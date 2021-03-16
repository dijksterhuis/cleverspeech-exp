# Log Probs Difference

As per CumulativeLogProbs, uses the alpha and beta variables from the Alex Graves CTC paper but with
a pre-determined target alignment that we find before running the attack.

These target log probabilities are then compared to the log probability of the current argmax of the
current softmax outputs.The difference between the two is the `max target - max others` equation
from Biggio et al.

Making `kappa >>> 1.0` can help the optimisation focus on making the target alignment most likely.

Aims to maximise the probability of a given alignment compared to the greedy alignment.