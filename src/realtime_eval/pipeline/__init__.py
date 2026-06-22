"""Orchestration layer that produces and analyzes sweep results.

Depends on :mod:`realtime_eval.core`:

- :mod:`realtime_eval.pipeline.runner`: warm, repeated, timed inference loop.
- :mod:`realtime_eval.pipeline.sweep`: cartesian grid -> results.jsonl + summary.
- :mod:`realtime_eval.pipeline.analyze`: load results -> table + best pick.
"""

from __future__ import annotations
