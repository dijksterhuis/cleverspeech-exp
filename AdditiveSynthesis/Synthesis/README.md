# Synthesis module

### Base
Base class that provides helper methods to the rest of the synthesis classes.
Eventually there will be more work here to avoid boilerplate elsewhere. 

### Additive
Contains the `Inharmonic`, `FreqHarmonic` and `FullyHarmonic` classes.
Used in the _Adversaries Can Add_ experiments.

**A lot** of the code in this module was based on the work by [magneta/ddsp][0].
Any methods that are directly derived from their work is indicated as such in the method docstring.

TODO: Not sure how licensing works for cases where derivatives works have changed a lot of the code.
Might be safest to assume that the `Additive` module is covered under the Apache 2 license.

### DeterministicPlusNoise
Basically an additive synthesiser combined with either NoSynth or Spectral Synthesis.

### Plain
Optimise the perturbation directly (NoSynth).

### Spectral
Inverse STFT synthesis.

[0]: https://github.com/magenta/ddsp