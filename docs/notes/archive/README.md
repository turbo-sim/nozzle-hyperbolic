# Historical repository audits

These documents retain the original source-review evidence and reasoning from
the October 2026 branch comparison and structural restructuring. Their counts,
paths, defects, pending edits and proposed APIs refer to their recorded revision.
They are not current implementation instructions.

| Record | Preserved scope |
|---|---|
| [Alberto repository audit](alberto_repository_audit.md) | Original Alberto/Andrea source comparison, inherited issues and baseline test results |
| [JAX implementation audit](jax_implementation_audit.md) | Original recovery sensitivities, stencil assumptions, differentiability restrictions and explicit-integration issues |
| [Code organization](code_organization.md) | Initial restructuring rationale, branch flux comparisons and state-representation distinctions |

Use the [active documentation index](../README.md) for the supported package.
The audits contain unresolved review topics for the deferred two-fluid model;
these are retained in the [research backlog](../proposals/research_backlog.md)
without assuming that every historical issue is still present or already fixed.

Historical Newton and flux-regularity explanations now live with their
[retained experiment](../../../development/newton_solve/README.md).
Redundant redirect notes and the old commit-staging diary were removed; their
original versions remain available in Git history.
