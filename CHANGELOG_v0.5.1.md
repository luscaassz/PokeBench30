# PokéBench30 v0.5.1

## Representation

- Introduces **UBR 1.2**.
- Observation dimension changes from **472 → 518**.
- Action dimension remains **10**.
- UBR 1.1's 472-feature core is preserved as an exact prefix.
- Schema SHA-256: `d4b3858aab3a1c3bb1e17d9f34becf93814be05bfc949d52023651035f20e4c3`.

## Added battle context

For both sides:

- Stealth Rock
- Spikes layers
- Toxic Spikes layers
- Sticky Web
- Reflect
- Light Screen
- Aurora Veil
- Tailwind

Global/public battle context:

- rain, sun, sandstorm, hail, snow
- Primordial Sea, Desolate Land, Delta Stream
- Electric, Grassy, Misty and Psychic Terrain
- Trick Room
- Gravity

For both active Pokémon:

- Taunt
- Encore
- Substitute
- Confusion
- Leech Seed
- Disable
- Yawn
- Protect

## RL

- MaskablePPO configuration is intentionally unchanged.
- Reward is intentionally unchanged.
- Old UBR 1.1 model metadata is rejected by the UBR 1.2 loader.
- Added TensorBoard to `requirements.txt` and the setup checker after the
  v0.5 live smoke test exposed the missing dependency.

## Validation

- Context features are range/binary checked.
- Offline tests verify hazard layers, weather, terrain, Trick Room and active
  volatile effects.
- The strict universal-state validator now checks battle-context values.
