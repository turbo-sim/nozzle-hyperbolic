# Shock tube

Run `poetry run python demos/shock_tube/run.py`. Edit physical inputs, recording
times and video controls at the top of [run.py](run.py).

A stationary diaphragm separates air at pressure ratio 10 and density ratio 8.
MUSCL/HLL with RK3 resolves the rarefaction, contact and shock. The default
continues to about 0.06685 s so the waves leave the one-metre domain.

The approximately 30-second animation uses 601 actual solved states at uniformly
spaced physical times and 20 fps. It has no initial or final pause by default;
each frame advances physical time by the same amount. Both MP4 and GIF are saved.
An independent Riemann reference is shown throughout. The right shock reaches
a boundary first at about 0.90 ms.
Later the unbounded reference is shown in grey as a comparison.

`reference_comparison.png/.pdf` compares three early times. Numerical/reference
arrays, diagnostic histories, publication figures, `transient.mp4` and
`transient.gif` are in `outputs/`.
See the [shared guide](../README.md) for caching and FFmpeg.
