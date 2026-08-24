# NEXUS Learn V5
Termux-first autonomous logarithm learning prototype.

V5 replaces the broken fixed 64-class answer head with a neural numerical regressor.
It is intentionally small and uses NumPy only. The learner starts from random weights.
The generator creates procedural logarithm problems; hidden evaluation uses a separate seed.
This is a research prototype, not a general LLM or AGI.


## 1.4.0 repair notes

The regression was in the learning path, not the evaluator. The previous regressor
trained directly against raw numerical targets, including values as large as
100000. With a small fixed learning rate this caused unstable gradients and
collapsed predictions, while the character-hash representation provided weak
numeric structure.

The repaired path:
- keeps NumPy as the only backend;
- learns `log1p(answer)` and maps predictions back with `expm1`, stabilizing the
  numerical target range;
- adds deterministic numeric/structural features alongside lexical features;
- uses fresh hidden data for evaluation;
- uses held-out failures only to select structural families for fresh training
  examples, preserving the hidden answers;
- reports actual measured accuracy and positive improvement rather than forcing
  benchmark output.

Regression result on this uploaded state:
- tests: 5 passed
- benchmark accuracy: 22.5%
- autonomous hidden accuracy: 0.0% -> 28.0%
- best hidden accuracy: 28.0%
- training examples: 1200
