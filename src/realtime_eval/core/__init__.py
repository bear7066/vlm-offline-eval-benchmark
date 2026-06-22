"""Reusable building blocks for the real-time sweep.

Pure, orchestration-free pieces shared across the pipeline:

- :mod:`realtime_eval.core.config`: sweep configuration and default prompt.
- :mod:`realtime_eval.core.metrics`: result dataclasses and aggregation.
- :mod:`realtime_eval.core.power`: background GPU power sampler.
- :mod:`realtime_eval.core.dataset`: labeled-video discovery.
"""

from __future__ import annotations
