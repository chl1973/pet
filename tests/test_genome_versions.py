"""基因组版本兼容：基因表只追加，旧 DNA 解码出的旧参数逐位不变，缺的新基因取中性值。

这里用一个假想的"第 2 版"基因表（临时追加两个基因）来检验，不改真正的 SCHEMA。
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                    # noqa: E402
import pytest                         # noqa: E402
import genome                         # noqa: E402
import persist                        # noqa: E402
from genome import NM                 # noqa: E402
from world import World, TPD          # noqa: E402
from creature import Creature         # noqa: E402

C1 = (ROOT / "c1.dna").read_text().strip()
C1_0 = (ROOT / "c1_0.dna").read_text().strip()

# 第 1 版解码结果的指纹（转接时的代码算出）。这两行变了，就说明 C1 已经不是 C1 了。
GOLDEN = {
    "c1.dna": ("c8be186120dad8ec614696dd23fcbdbe820046c73601a128ade84be2c371d90a", 1937018140),
    "c1_0.dna": ("32faf9b0a81f661b6e8108bacb691be19973c90efab969cfe4d4e8f2c5b21fb6", 2974582279),
}
V1_NAMES = [e[0] for e in genome.SCHEMA[:genome.VERSION_ENDS[0]]]

FAKE_V2 = [
    ("U_new_sense", (NM, 2), -1.2, 1.2, 0.0),   # 新感官 -> 化学物质（权重类，中性值 0）
    ("hand_cost", (), 0.0, 3.0, 1.5),            # 某个非权重参数，中性值取旧行为对应的值
]


def fingerprint(p, names):
    h = hashlib.sha256()
    for n in names:
        h.update(n.encode())
        h.update(np.asarray(p[n], dtype=np.float64).tobytes())
    return h.hexdigest()


@pytest.fixture
def v2(monkeypatch):
    schema = genome.SCHEMA + FAKE_V2
    monkeypatch.setattr(genome, "SCHEMA", schema)
    monkeypatch.setattr(genome, "VERSION_ENDS", genome.VERSION_ENDS + [len(schema)])
    genome.check_schema()
    return genome


@pytest.mark.parametrize("fname", list(GOLDEN))
def test_v1_decode_is_bit_identical_to_handoff(fname):
    dna = C1 if fname == "c1.dna" else C1_0
    p = genome.decode(dna)
    assert (fingerprint(p, V1_NAMES), p["seed"]) == GOLDEN[fname]


@pytest.mark.parametrize("fname", list(GOLDEN))
def test_old_dna_under_new_schema(v2, fname):
    dna = C1 if fname == "c1.dna" else C1_0
    assert v2.version_of(dna) == 1 and v2.schema_version() == 2
    p = v2.decode(dna)
    assert (fingerprint(p, V1_NAMES), p["seed"]) == GOLDEN[fname]     # 旧参数和个体种子逐位不变
    assert p["U_new_sense"].shape == (NM, 2) and np.all(p["U_new_sense"] == 0.0)
    assert p["hand_cost"] == 1.5
    assert p["dna_version"] == 1


def test_upgrade_is_lossless(v2):
    up = v2.upgrade(C1)
    assert len(up) == v2.n_genes() * 4 and up.startswith(C1)
    assert v2.version_of(up) == 2 and v2.canonical(up) == C1
    a, b = v2.decode(C1), v2.decode(up)
    for name, *_ in v2.SCHEMA:
        assert np.array_equal(a[name], b[name])
    assert a["seed"] == b["seed"]
    assert v2.upgrade(up) == up


def test_new_genes_span_their_range(v2):
    """新基因：GAAA 正好是中性值，AAAA 是下限，TTTT 是上限，并且单调。"""
    base = v2.upgrade(C1)
    k = v2.n_genes(1) * 4          # 第一个新基因的位置
    vals = []
    for code in ["AAAA", "CTTT", v2.NEUTRAL_CODON, "GAAC", "TTTT"]:
        vals.append(v2.decode(base[:k] + code + base[k + 4:])["U_new_sense"].ravel()[0])
    assert vals[0] == -1.2 and vals[2] == 0.0 and vals[-1] == 1.2
    assert all(x < y for x, y in zip(vals, vals[1:]))


def test_evolution_operators_upgrade_old_dna(v2):
    m = v2.mutate(C1, 0.05, 1)
    x = v2.crossover(C1, C1_0, 2)
    for d in (m, x):
        assert v2.version_of(d) == 2
        p = v2.decode(d)
        for name, shape, lo, hi, *_ in v2.SCHEMA:
            v = np.asarray(p[name])
            assert v.min() >= lo - 1e-12 and v.max() <= hi + 1e-12
    r = v2.random_dna(5)
    assert v2.version_of(r) == 2


def test_c1_lives_the_same_life_under_new_schema(v2):
    """新基因取中性值、且身体还没用到它们时，C1 的一生和第 1 版逐 tick 相同；补齐后的 DNA 也一样。"""
    def life(dna):
        w, c = World(7), Creature("C1", dna)
        for _ in range(TPD):
            w.step()
            c.step(w)
        return persist.dumps(w, c).replace(c.dna, "")   # 去掉 DNA 字符串本身再比
    ref = life(C1)
    assert life(v2.upgrade(C1)) == ref


def test_save_from_old_dna_loads_with_upgraded_dna(v2, tmp_path):
    w, c = World(7), Creature("C1", C1)
    for _ in range(50):
        w.step()
        c.step(w)
    sd = c.state_dict()
    c2 = Creature("C1", v2.upgrade(C1))
    c2.load_state_dict(sd)                     # 规范形式相同，认作同一个个体
    with pytest.raises(AssertionError):
        Creature("x", v2.upgrade(C1_0)).load_state_dict(sd)


def test_bad_dna_is_rejected():
    with pytest.raises(ValueError):
        genome.decode(C1[:-4])
    with pytest.raises(ValueError):
        genome.decode(C1[:-1] + "N")
    with pytest.raises(ValueError):
        genome.decode(C1 + "\n")


def test_schema_rules_are_enforced(monkeypatch):
    v1 = genome.SCHEMA
    monkeypatch.setattr(genome, "SCHEMA", v1 + [("no_neutral", (), 0.0, 1.0)])
    monkeypatch.setattr(genome, "VERSION_ENDS", genome.VERSION_ENDS + [len(v1) + 1])
    with pytest.raises(AssertionError):
        genome.check_schema()
    monkeypatch.setattr(genome, "SCHEMA", v1 + [("bad", (), 0.0, 1.0, 2.0)])
    with pytest.raises(AssertionError):
        genome.check_schema()
