"""Real-time VLM evaluation: find the largest model config that stays real time.

See ``PLAN.md`` for the design. Public surface:

- :class:`realtime_eval.core.config.SweepConfig`
- :class:`realtime_eval.core.metrics.RealtimeResult`
- :func:`realtime_eval.pipeline.runner.run_config`
- :func:`realtime_eval.pipeline.sweep.run_sweep`
"""

from __future__ import annotations

__all__ = [
    "SweepConfig",
    "RealtimeResult",
]
