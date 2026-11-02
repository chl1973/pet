"""核心回归测试。改任何东西之后跑一遍：python -m pytest -q"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                     # noqa: E402
import pytest                                          # noqa: E402
from genome import (random_dna, decode, mutate, crossover, n_genes, version_of,  # noqa: E402
                    SCHEMA, NS, NM, NA, SENSORS, SENSOR_GROUP)
from world import World, TPD, DEVICES, digest, MESS_W  # noqa: E402
from creature import Creature                          # noqa: E402

C1 = (ROOT / "c1.dna").read_text().strip()


# ---------- 基因组 ----------
def test_c1_genome_matches_schema():
    assert version_of(C1) == 1
    p = decode(C1)
    for name, shape, lo, hi, *_ in SCHEMA:
        v = np.asarray(p[name])
        assert v.shape == (shape if shape else ())
        assert v.min() >= lo - 1e-9 and v.max() <= hi + 1e-9


def test_mutate_and_crossover_keep_length():
    a, b = random_dna(1), random_dna(2)
    assert len(mutate(a, 0.1, 3)) == len(a)
    assert len(crossover(a, b, 4)) == len(a)
    assert decode(a)["seed"] == decode(a)["seed"]


def test_sensor_table_consistent():
    assert len(SENSORS) == len(SENSOR_GROUP) == NS


# ---------- 世界规则（保守序列） ----------
def test_digest_only_pays_for_real_reduction():
    assert digest(10.0, 5.0) == pytest.approx(30.0)
    assert digest(5.0, 10.0) == 0.0          # 让世界更乱不给能量
    assert digest(5.0, 5.0) == 0.0


def test_counts_never_negative_and_no_hard_delete():
    w, c = World(7), Creature("t", C1)
    for _ in range(3 * TPD):
        w.step()
        c.step(w)
        for d in DEVICES:
            assert all(v >= 0 for v in w.h[d].c.values())
    # 有用文件只会在 ok / misplaced / missing 之间移动，误吞的都还在回收堆里可恢复
    assert w.log["compost_valuable"] >= w.log["restored"]


def test_user_behaviour_independent_of_creature():
    """有没有生命体，用户的作息和产出必须完全一样（对照组可比的前提）。"""
    w1, w2 = World(7), World(7)
    c = Creature("t", C1)
    for _ in range(2 * TPD):
        w1.step()
        c.step(w1)
        w2.step()
        for d in DEVICES:
            assert w1.h[d].user == w2.h[d].user
            assert w1.h[d].online == w2.h[d].online


# ---------- 生命体 ----------
def test_simulation_is_deterministic():
    def trace():
        w, c = World(8), Creature("t", C1)
        out = []
        for _ in range(TPD):
            w.step()
            c.step(w)
            out.append(c.E)
        return np.array(out)
    assert np.array_equal(trace(), trace())


def test_reward_comes_only_from_hunger_and_pain():
    c = Creature("t", C1)
    c.E, c.damage = c.Emax(), 0.0
    assert c.drive() == pytest.approx(0.0)
    c.E = 0.5 * c.Emax()
    assert c.drive() > 0
    c.E, c.damage = c.Emax(), 0.5
    assert c.drive() > 0


def test_c1_survives_unseen_world():
    """演化出的 C1 在没见过的世界 7 里活 7 天不应该结孢子。"""
    w, c = World(7), Creature("C1", C1)
    for _ in range(7 * TPD):
        w.step()
        c.step(w)
    assert c.spore_count == 0
    assert c.E / c.Emax() > 0.3


def test_chemicals_stay_bounded():
    w, c = World(9), Creature("t", C1)
    for _ in range(2 * TPD):
        w.step()
        c.step(w)
        assert np.all(np.abs(c.m) <= 1.0 + 1e-9)
        assert c.m.shape == (NM,)
