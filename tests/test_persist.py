"""存档与恢复：中途存档、重新加载后接着跑，必须和不中断时逐 tick 完全一致。

比较的不是几个摘要数字，而是每个 tick 世界和生命体的完整存档（包括随机数状态、
行为网络权重、资格迹、化学浓度、发育与表观遗传标记），逐字节相同。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                    # noqa: E402
import pytest                         # noqa: E402
import persist                        # noqa: E402
from genome import random_dna         # noqa: E402
from world import World, TPD          # noqa: E402
from creature import Creature         # noqa: E402

C1 = (ROOT / "c1.dna").read_text().strip()
SPORER = random_dna(6)   # 随机基因组，在世界 7 里第 59 tick 结孢子、第 137 tick 复苏，用来覆盖孢子期


def fingerprints(w, c, ticks):
    out = []
    for _ in range(ticks):
        w.step()
        if c is not None:
            c.step(w)
        out.append(persist.dumps(w, c))
    return out


def run_with_restarts(dna, wseed, ticks, cut_at, path, learn=True):
    """在 cut_at 的每个时刻存档到文件、丢掉内存里的对象、再从文件读回来接着跑。"""
    w = World(wseed)
    c = Creature("C1", dna, learn=learn) if dna else None
    out, t = [], 0
    for cut in list(cut_at) + [ticks]:
        out += fingerprints(w, c, cut - t)
        t = cut
        if t < ticks:
            persist.save(path, w, c)
            del w, c
            w, c = persist.load(path)
    return out


@pytest.mark.parametrize("dna,wseed,ticks,cuts,learn", [
    (C1, 7, 2 * TPD, [TPD + 37], True),                # C1 成长期，敏感期内
    (C1, 8, 2 * TPD, [1, 100, 101, 400], True),        # 多次重启，包括出生后第一个 tick
    (C1, 9, TPD, [150], False),                        # 无学习兄弟
    (SPORER, 7, 200, [58, 59, 100, 137], True),        # 结孢子前后、孢子期中、复苏时
    (None, 7, TPD, [123], True),                       # 空白世界（对照组）
])
def test_resume_matches_uninterrupted_tick_by_tick(tmp_path, dna, wseed, ticks, cuts, learn):
    ref = fingerprints(World(wseed), Creature("C1", dna, learn=learn) if dna else None, ticks)
    got = run_with_restarts(dna, wseed, ticks, cuts, tmp_path / "c1.save", learn=learn)
    assert len(got) == len(ref)
    bad = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), None)
    assert bad is None, f"第 {bad} 个 tick 开始和不中断时不一致"


def test_resume_after_critical_period_and_epigenetics(tmp_path):
    """敏感期结束、表观遗传标记写下之后存档，读回来标记和后续轨迹都不变。"""
    w, c = World(7), Creature("C1", C1)
    end = int(np.ceil(c.p["cp_end_days"] * TPD)) + 5
    fingerprints(w, c, end)
    assert c.epi is not None
    persist.save(tmp_path / "s", w, c)
    w2, c2 = persist.load(tmp_path / "s")
    assert c2.epi == c.epi and np.array_equal(c2.U_mult, c.U_mult) and c2.pain_mult == c.pain_mult
    assert fingerprints(w, c, 100) == fingerprints(w2, c2, 100)


def test_sporer_actually_covers_spore_and_revival():
    w, c = World(7), Creature("x", SPORER)
    events = set()
    for _ in range(200):
        w.step()
        c.step(w)
        events.add(c.event)
    assert {"spore", "revive"} <= events


def test_state_dict_is_a_snapshot_not_a_view():
    w, c = World(7), Creature("C1", C1)
    fingerprints(w, c, 50)
    sw, sc = w.state_dict(), c.state_dict()
    before = persist.to_tree(dict(w=sw, c=sc))
    fingerprints(w, c, 50)
    assert persist.to_tree(dict(w=sw, c=sc)) == before


def test_types_survive_round_trip():
    x = dict(a=np.float64(0.1), b=0.1, c=np.arange(6, dtype=np.int64).reshape(2, 3),
             d=("laptop", 3), e=[1, (2.5, None)], f=np.bool_(True), g=True, h=-10 ** 9)
    y = persist.from_tree(persist.to_tree(x))
    assert type(y["a"]) is np.float64 and type(y["b"]) is float
    assert y["c"].dtype == np.int64 and y["c"].shape == (2, 3)
    assert y["d"] == ("laptop", 3) and isinstance(y["e"][1], tuple)
    assert type(y["f"]) is np.bool_ and y["g"] is True and y["h"] == -10 ** 9
    with pytest.raises(TypeError):
        persist.to_tree(dict(f=lambda: 0))


def test_load_refuses_wrong_genome_and_incomplete_save():
    c = Creature("C1", C1)
    sd = c.state_dict()
    with pytest.raises(AssertionError):
        Creature("x", random_dna(3)).load_state_dict(sd)
    del sd["za_W1"]
    with pytest.raises(AssertionError):
        Creature("C1", C1).load_state_dict(sd)
