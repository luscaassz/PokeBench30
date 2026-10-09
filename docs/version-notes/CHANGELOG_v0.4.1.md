# PokéBench30 v0.4.1 changelog

## Fixed

- UBR move slots now represent transient Gen I legal commands such as `recharge` and `fight`.
- Persistent move ordering is preserved in ordinary states.
- Transient legal moves replace only currently unavailable move slots when all four persistent slots are occupied.
- The action-space implementation now fails loudly if more than four legal moves are ever exposed simultaneously.
- Dataset validation now fails on unmapped chosen actions.
- Dataset validation now fails on empty legal-action masks.
- Validator diagnostics cross-reference `actions.csv` when available.

## Versioned representation

- Project version: `0.4.1`.
- Representation version: `ubr-1.1`.
- Observation dimension remains `472`.
- Action dimension remains `10`.
- Schema SHA-256 changes because move-slot semantics changed.

## Regression tests

Added explicit tests for:

- `recharge` with four persistent moves;
- `fight` with four persistent moves;
- preservation of ordinary persistent move ordering.

The previously collected v0.4 smoke-test dataset is expected to fail the new strict validator because it contains 8 unmapped Gen I decisions.
