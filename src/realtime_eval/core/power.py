from __future__ import annotations

import threading
import time

from vlm_eval.hardware import get_gpu_power_watts


class PowerSampler:
    """Background sampler for GPU power draw during an inference.

    Replaces the crude two-sample (start/end) average used by
    ``vlm_eval``: a daemon thread polls ``nvidia-smi`` at a fixed interval
    while the sampler is active, then exposes the mean and peak.

    Use it as a context manager around the work to be measured::

        with PowerSampler(interval_sec=0.1) as sampler:
            model.generate(...)
        print(sampler.mean_watts, sampler.peak_watts)

    Args:
        interval_sec: Seconds between successive power readings.
    """

    def __init__(self, interval_sec: float = 0.1) -> None:
        self.interval_sec = interval_sec
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            watts = get_gpu_power_watts()
            if watts is not None:
                self._samples.append(watts)
            self._stop.wait(self.interval_sec)

    def __enter__(self) -> "PowerSampler":
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        # Guarantee at least one reading even for very short inferences.
        if not self._samples:
            watts = get_gpu_power_watts()
            if watts is not None:
                self._samples.append(watts)

    @property
    def mean_watts(self) -> float | None:
        """Mean power draw across all samples, or ``None`` if none collected."""
        if not self._samples:
            return None
        return sum(self._samples) / len(self._samples)

    @property
    def peak_watts(self) -> float | None:
        """Maximum power draw across all samples, or ``None`` if none collected."""
        if not self._samples:
            return None
        return max(self._samples)
