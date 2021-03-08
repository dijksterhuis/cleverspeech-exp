# Alignment Edge Cases

**Main idea:** what happens during attacks when we focus on different types of alignments.

```
Sparse Alignment  = o----------p---------e-------------n------------
Dense Alignment   = oooooooooooppppppppppeeeeeeeeeeeeeennnnnnnnnnnnn
```
- Does one type produce more confident adversarial examples than the other?
- Is one more robust to destructive transforms than the other?
- Which one achieves the smallest distance?
- What happens to confidence as distance is minimised?

Additional: can we find the most optimal alignment for a model using CTC loss?

### Additional alignment case: best CTC alignment for a target (independent of an example)

A confident sparse alignment can be derived from the argmax of a dummy softmax matrix (initially
zero valued) by solving the following optimisation goal:

```
minimise_{Z} CTC Loss(Z, t')
```

By optimising a dummy matrix, we end up with a matrix where the most likely characters per frame are
(usually) the targets we want.

*Most importantly*, this optimisation is independent of any audio example -- the alignment we find
will be the most likely alignment for _this_ model and for _this_ target transcription.


### Tricking CTC Loss

CTC Loss is used to perform maximum likelihood optimisation over **all** valid alignments, and is
usually used with respect to a target transcription of length less than or equal to the number of
audio frames.

But we can trick CTC Loss into optimising an alignment with length equal to the number of audio
frames with two key changes.

The below settings are an edge case for CTC Loss use in tensorflow:
```python
preprocess_collapse_repeated=False,
ctc_merge_repeated=True,
```

Enabling them means that:
- (a) duplicated characters in a target sequence are not collapsed
- (b) duplicated characters are not merged/removed during optimisation

We can't pass in an alignment sequence as a target without tricking CTC Loss into treating the
blank token `-` as a real/valid character.

The network's softmax matrix output is extended with an M (n_frames) length vector of zero values.
These extra values act as a "dummy" blank token, which will never be likely.

Now we can include the blank `-` token in the target sequence. This obviously modifies one of the
conditions for CTC Loss -- now the target sequence length must be equal to the number of audio
frames.
