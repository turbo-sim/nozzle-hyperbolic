# Preserved production baselines

[`pre_generic_fvm/`](pre_generic_fvm/README.md) contains the production source
and tests from before the shared finite-volume integration. It is the native
baseline for comparisons with the current implementation in
[`src/nozzle_hyperbolic/`](../src/nozzle_hyperbolic/).

The preserved package stays outside `src/` and normal production imports.
The [native performance study](../development/roberto/nozzle_native_performance/README.md)
loads it explicitly in a separate worker process, using its original package
namespace. The benchmark's `ARCHIVE` constant identifies the selected tree.

Treat snapshot source and tests as frozen. Verify them against
[`source_manifest.json`](pre_generic_fvm/source_manifest.json) before timing;
do not modify the baseline to simplify a comparison. Keep benchmark adapters,
configuration and reports in the performance study, outside the snapshot.

Moving this directory does not change the source identity. Existing saved
results retain their measurement-time paths and hashes; a new timing campaign
must record the source and settings it actually executes.
