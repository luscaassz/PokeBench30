# PokéBench30 Battle Lab — MVP

A functional, no-build web application for exploring the **10-seed cross-generation transfer + telemetry study**.

This MVP is intentionally implemented with plain HTML/CSS/JavaScript so it runs immediately without npm, Node packages, or an internet connection.

## Included

- Pixel-art / handheld-RPG-inspired visual identity.
- Overview screen with in-domain scorecards and behavioral fingerprint.
- Interactive 3×3 transfer matrix.
- Transfer-gap explorer.
- Telemetry explorer:
  - damaging/status/buff/debuff behavior,
  - STAB / super-effective / resisted rates,
  - top moves,
  - selection-given-availability,
  - move types,
  - battle dynamics.
- Generation DNA:
  - species/move counts,
  - dual typing,
  - damaging/status composition,
  - fixed-pool tactical features.
- Battle Inspector using the **real frozen fixed teams**.
- Reproducible exporter:
  `scripts/export_battle_lab_data.py`.

## Important Battle Inspector limitation

`aggregate(5).zip` contains aggregate summaries and frozen-team catalogs, but **not the raw per-turn `actions.csv` files**.

The Battle Inspector therefore shows:

- real fixed teams,
- real train→test behavioral profile,
- real aggregate battle dynamics,

but **does not fabricate a turn-by-turn replay**.

A future v2 can unlock replay once the raw evaluation folders/actions logs are supplied.

## Run on Windows

Open PowerShell inside this folder:

```powershell
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

You can also double-click `START_BATTLE_LAB.bat`.

## Refresh data from a future aggregate

```powershell
python scripts\export_battle_lab_data.py "C:\path\to\aggregate.zip" --out data\battle_lab_data.json
```

Then reload the browser.

## Current source

- Data: `aggregate(5).zip`
- SHA-256: `fcb4a87b853face45bc8f7632f7f94f7d7788f3c1bb8eaa8c872d1262bd27216`
- Seeds: 10
- Controlled transfer battles represented: 10800

## Suggested next version

The strongest next feature is a **true Battle Replay** driven by raw `actions.csv` and per-battle logs:

- turn slider,
- actual active Pokémon,
- chosen move/switch,
- legal alternatives,
- HP/status before and after,
- telemetry annotations,
- side-by-side policy comparison.
