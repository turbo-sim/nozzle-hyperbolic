# Shock tube

Run ?poetry run python demos/shock_tube/run.py?. Edit physical inputs, recording
times and video controls at the top of [run.py](run.py).

A stationary diaphragm separates air at pressure ratio 10 and density ratio 8.
MUSCL/HLL with RK3 resolves the rarefaction, contact and shock. The default
continues to about 0.06685 s so the waves leave the one-metre domain.

The 35-second movie uses actual saved states and an independent Riemann
reference. Forty percent of its moving frames show early waves before boundary
interaction. The right shock reaches a boundary first at about 0.90 ms.
Later the unbounded reference is shown in grey as a comparison.

?reference_comparison.png/.pdf? compares three early times. Numerical/reference
arrays, diagnostic histories, publication figures and MP4 are in ?outputs/?.
See the [shared guide](../README.md) for caching and FFmpeg.
