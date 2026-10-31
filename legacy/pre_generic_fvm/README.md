# Source snapshot before the shared FVM port

This directory preserves the previous production source and tests for review
and recovery. `source_manifest.json` records the snapshot checksums. Additional
retired compatibility files are under `retired_compatibility/`.

These are historical files, not another supported solver or import target.
The current implementations live in
[`src/nozzle_hyperbolic`](../../src/nozzle_hyperbolic/). Current verification
uses [`tests/`](../../tests/) and the
[transient study](../../development/roberto/transient_problems/README.md).
The [native benchmark](../../development/roberto/nozzle_native_performance/README.md)
loads this frozen source explicitly; see the [baseline policy](../README.md).
The [archived migration guide](../../docs/old/MIGRATION.md) records the API
changes and deliberately deferred two-fluid migration.
