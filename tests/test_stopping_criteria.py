import time

from src.stopping_criteria import (
    MaxGenerations, TimeLimit, GenMinImprovement, TimeMinImprovement, TargetObjective,
)


def test_max_generations():
    c = MaxGenerations(5)
    c.reset()
    for i in range(4):
        assert not c.check([100 - i])
    assert c.check([96])
    assert c.triggered is True


def test_time_limit():
    c = TimeLimit(0.1)
    c.reset()
    assert not c.check([100])
    time.sleep(0.15)
    assert c.check([100, 99])
    assert c.triggered is True


def test_gen_min_improvement_triggers():
    c = GenMinImprovement(window=3, min_pct=0.05)
    c.reset()
    history = [100, 100, 100]
    assert c.check(history)
    assert c.triggered is True


def test_gen_min_improvement_no_trigger():
    c = GenMinImprovement(window=3, min_pct=0.05)
    c.reset()
    history = [100, 90, 80]
    assert not c.check(history)


def test_gen_min_improvement_short_history():
    c = GenMinImprovement(window=5, min_pct=0.05)
    c.reset()
    assert not c.check([100])
    assert not c.check([100, 99])


def test_time_min_improvement_triggers():
    c = TimeMinImprovement(window=0.1, min_pct=0.05)
    c.reset()
    c.check([100])
    time.sleep(0.15)
    assert c.check([100, 100])
    assert c.triggered is True


def test_time_min_improvement_no_trigger():
    """If there's real improvement, should not trigger."""
    c = TimeMinImprovement(window=0.1, min_pct=0.01)
    c.reset()
    c.check([100])
    time.sleep(0.15)
    # 50% improvement — should not trigger
    assert not c.check([100, 50])


def test_target_objective():
    c = TargetObjective(target=50)
    c.reset()
    assert not c.check([100])
    assert c.check([100, 50])
    assert c.triggered is True


def test_triggered_resets():
    c = MaxGenerations(2)
    c.reset()
    c.check([100])
    c.check([100, 99])
    assert c.triggered is True
    c.reset()
    assert c.triggered is False


def test_gen_counter():
    c = MaxGenerations(10)
    c.reset()
    assert c.gen == 0
    c.check([100])
    assert c.gen == 1
    c.check([100, 99])
    assert c.gen == 2
    c.reset()
    assert c.gen == 0
