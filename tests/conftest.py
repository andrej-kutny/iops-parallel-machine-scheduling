import os
import pytest

from src.models.instance import SchedulingInstance
from src.models.solution import SchedulingSolution


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "data")
SMALL_INSTANCE_PATH = os.path.join(DATA_DIR, "75_3_5_H.json")


@pytest.fixture
def small_instance():
    return SchedulingInstance(SMALL_INSTANCE_PATH)


@pytest.fixture
def small_instance_path():
    return SMALL_INSTANCE_PATH


@pytest.fixture
def valid_schedule(small_instance):
    """A known valid schedule for the 5-job, 3-machine instance.
    Jobs 1-4 can only run on machine 2. Job 5 can run on 0, 1, or 2.
    """
    schedule = {
        0: [],
        1: [5],
        2: [2, 3, 1, 4],
    }
    return SchedulingSolution(schedule, small_instance)
