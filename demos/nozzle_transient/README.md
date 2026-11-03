# Nozzle pressure release

Run ?poetry run python demos/nozzle_transient/run.py?. Edit physical inputs,
mesh, end time and video controls at the top of [run.py](run.py).

Air starts at rest at 2 bar and 300 K. The exterior suddenly drops to 0.02 bar.
A finite-wave outlet law selects pressure, sonic or outward-only branches.

MUSCL/HLL with RK3 runs for eight acoustic transit times. The 35-second movie
allocates 60% of its moving frames to the first two transit times. Fixed axes,
physical-time labels and initial/final holds make start-up easier to follow.

Full states, diagnostic histories, publication figures and MP4 are in ?outputs/?.
See the [shared guide](../README.md) for caching and FFmpeg.
