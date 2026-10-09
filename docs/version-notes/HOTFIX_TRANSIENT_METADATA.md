# v0.4.1 hotfix — Gen I transient move metadata

Live Gen I testing showed that poke-env may expose pseudo-moves such as
`recharge` and `fight` with incomplete internal move dictionaries. Some
properties, especially `priority`, raise `KeyError` instead of returning a
default.

The corrected build:
- safely handles missing transient move metadata;
- keeps UBR 1.1 at 472 observation features and 10 actions;
- maps transient moves without probing optional move properties;
- hardens TypeAware scoring and action logging;
- adds regression tests that deliberately raise `KeyError`.

UBR remains 1.1. This is an implementation robustness fix, not a schema-layout
change.
