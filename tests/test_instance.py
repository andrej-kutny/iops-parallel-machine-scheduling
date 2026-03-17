import numpy as np


def test_instance_loads(small_instance):
    assert small_instance.n == 5
    assert small_instance.m == 3
    assert small_instance.horizon == 1664


def test_instance_shapes(small_instance):
    assert small_instance.duration.shape == (5, 3)
    assert small_instance.release.shape == (5, 3)
    assert small_instance.setup.shape == (5, 5, 3)


def test_capable_lists(small_instance):
    assert small_instance.capable[0] == [2]
    assert small_instance.capable[1] == [2]
    assert small_instance.capable[2] == [2]
    assert small_instance.capable[3] == [2]
    assert set(small_instance.capable[4]) == {0, 1, 2}


def test_machine_capable_jobs(small_instance):
    mcj = small_instance.machine_capable_jobs
    assert 4 in mcj[0]  # job 5 (0-indexed: 4) can run on machine 0
    assert 4 in mcj[1]  # job 5 can run on machine 1
    assert len(mcj[2]) == 5  # all jobs can run on machine 2


def test_duration_values(small_instance):
    # Job 1 (index 0) on machine 2 has duration 352
    assert small_instance.duration[0][2] == 352


def test_repr(small_instance):
    r = repr(small_instance)
    assert "n=5" in r
    assert "m=3" in r
