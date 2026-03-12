import time
from abc import ABC, abstractmethod


class StoppingCriterion(ABC):
    """Abstract stopping criterion with triggered flag."""

    def __init__(self):
        self.triggered: bool = False
        self._checks: int = 0

    def reset(self):
        self.triggered = False
        self._checks = 0

    @abstractmethod
    def should_stop(self, history: list[float]) -> bool:
        ...

    def check(self, history: list[float]) -> bool:
        """Check and set triggered flag. Tracks generation count internally."""
        self._checks += 1
        result = self.should_stop(history)
        if result:
            self.triggered = True
        return result

    @property
    def gen(self) -> int:
        """Current generation (number of checks performed so far)."""
        return self._checks

    @abstractmethod
    def __repr__(self) -> str:
        ...


class MaxGenerations(StoppingCriterion):
    def __init__(self, n: int):
        super().__init__()
        assert n > 0, "n must be > 0"
        self.n = n

    def should_stop(self, history: list[float]) -> bool:
        return self._checks >= self.n

    def __repr__(self) -> str:
        return f"MaxGenerations({self.n})"


class TimeLimit(StoppingCriterion):
    def __init__(self, seconds: float):
        super().__init__()
        assert seconds > 0, "seconds must be > 0"
        self.seconds = seconds
        self._start: float | None = None

    def reset(self):
        super().reset()
        self._start = time.monotonic()

    def should_stop(self, history: list[float]) -> bool:
        if self._start is None:
            self._start = time.monotonic()
        return (time.monotonic() - self._start) >= self.seconds

    def __repr__(self) -> str:
        return f"TimeLimit({self.seconds}s)"


_DEFAULT_MIN_PCT = 0.01


class GenMinImprovement(StoppingCriterion):
    """Stop when improvement over a generation window is below threshold."""

    def __init__(self, window: int = 20, min_pct: float = _DEFAULT_MIN_PCT):
        super().__init__()
        assert window > 0, "window must be > 0"
        assert 0 <= min_pct < 1, "min_pct must be in [0, 1)"
        self.window = window
        self.min_pct = min_pct

    def should_stop(self, history: list[float]) -> bool:
        if len(history) < self.window:
            return False
        oldest = history[-self.window]
        current = history[-1]
        if oldest == 0:
            return False
        improvement = (oldest - current) / oldest
        return improvement <= self.min_pct

    def __repr__(self) -> str:
        return f"GenMinImprovement(window={self.window}, min_pct={self.min_pct})"


class TimeMinImprovement(StoppingCriterion):
    """Stop when improvement over a time window (seconds) is below threshold."""

    def __init__(self, window: float = 30.0, min_pct: float = _DEFAULT_MIN_PCT):
        super().__init__()
        assert window > 0, "window must be > 0"
        assert 0 <= min_pct < 1, "min_pct must be in [0, 1)"
        self.window = window
        self.min_pct = min_pct
        self._time_marks: list[float] = []

    def reset(self):
        super().reset()
        self._time_marks = []

    def should_stop(self, history: list[float]) -> bool:
        now = time.monotonic()
        self._time_marks.append(now)

        if len(history) < 2:
            return False

        current = history[-1]
        cutoff = now - self.window

        # Find the history index at the time cutoff
        oldest_idx = 0
        for i, t in enumerate(self._time_marks):
            if t >= cutoff:
                oldest_idx = max(0, i - 1)
                break

        if oldest_idx >= len(history):
            return False

        oldest = history[oldest_idx]
        if oldest == 0:
            return False
        improvement = (oldest - current) / oldest
        return improvement <= self.min_pct

    def __repr__(self) -> str:
        return f"TimeMinImprovement(window={self.window}s, min_pct={self.min_pct})"


class TargetObjective(StoppingCriterion):
    def __init__(self, target: float):
        super().__init__()
        assert target > 0, "target must be > 0"
        self.target = target

    def should_stop(self, history: list[float]) -> bool:
        return len(history) > 0 and history[-1] <= self.target

    def __repr__(self) -> str:
        return f"TargetObjective({self.target})"
