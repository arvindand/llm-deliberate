# Changelog

## 0.1.0 - 2026-10-31

First tagged research release, including the existing deliberation, voting and cost-analysis workflows.

- Add CI for the locked Python/backend tests, CLI/package build and frontend tests/build.
- Restore JSON experiment exports with `deliberate import path.json`, preserving response IDs, rounds, rankings and metadata. Existing experiments are never overwritten.
- Test the bundled no-key demo and export/import round trip, including rejected imports.
- Repair the bundled demo's stale round-zero markers to match its recorded responses and the current data model.
