# PokéBench30 v0.6.1 Hotfix 3 — non-finite UBR observation guard

## Trigger

During extension of the v0.6.0 cross-generation study from five to ten
independent seeds, Gen I seed 27182 stopped at 16,384 transitions with a
PyTorch `Categorical` error. One of the two shared-policy observation rows
produced all-NaN action logits while the other row remained finite.

Because both rows share the same policy parameters, this pattern is
consistent with a non-finite value in one live UBR observation rather than
global corruption of the PPO weights.

## Fix

- `representation/encoder.py`
  - `_clip` is now finite-safe.
  - NaN/Inf numeric battle values are mapped to zero before clipping.
  - move accuracy preserves its prior "unknown -> 1.0" convention for
    non-finite values.
  - final UBR output has a fail-fast finite invariant with feature names.
- `rl/wang_style.py`
  - policy feature extraction adds a final `torch.nan_to_num` transport
    guard as defense in depth.

No change is made to finite observations, the UBR dimensionality/schema,
action space, PPO hyperparameters, reward, fixed teams, or evaluation
protocol.

Completed models need not be retrained. The guard affects only a state that
would otherwise crash the run.
