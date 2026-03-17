import json
import numpy as np


class SchedulingInstance:
    """Parallel machine scheduling instance loaded from JSON."""

    def __init__(self, filepath: str):
        with open(filepath, "r") as f:
            data = json.load(f)

        self.n: int = data["n"]
        self.m: int = data["m"]
        self.horizon: int = data["horizon"]
        self.capable: list[list[int]] = data["capable"]
        self.duration: np.ndarray = np.array(data["duration"], dtype=int)
        self.release: np.ndarray = np.array(data["release"], dtype=int)
        self.setup: np.ndarray = np.array(data["setup"], dtype=int)

        self._machine_capable_jobs: dict[int, list[int]] | None = None

    @property
    def machine_capable_jobs(self) -> dict[int, list[int]]:
        """Inverse mapping: {machine_id: [job_0idx, ...]}."""
        if self._machine_capable_jobs is None:
            self._machine_capable_jobs = {k: [] for k in range(self.m)}
            for j in range(self.n):
                for k in self.capable[j]:
                    self._machine_capable_jobs[k].append(j)
        return self._machine_capable_jobs

    def __repr__(self) -> str:
        return f"SchedulingInstance(n={self.n}, m={self.m}, horizon={self.horizon})"
