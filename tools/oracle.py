"""检验世界本身是否可生存：用一个手写的"作弊"策略代替 C1 的大脑跑 30 天。

改了世界规则（world.py）或代谢常数（creature.py 顶部）之后先跑这个。
手写策略都活不下来，说明物理规则太苛刻，问题不在 C1。
这个策略只用于检验，绝不能用在 C1 身上。

用法: python tools/oracle.py [天数]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                     # noqa: E402
from genome import ACTIONS             # noqa: E402
from world import World, TPD, DEVICES  # noqa: E402
from creature import Creature          # noqa: E402

A = {n: i for i, n in enumerate(ACTIONS)}


def oracle(c, w):
    if c.loc != "nest":
        h = w.h[c.loc].c
        if h["temp"] + h["dup"] > 0:
            return A["eat_quick"]
        if h["misplaced"] > 0 and c.E > 5:
            return A["eat_careful"]
    best = max([d for d in DEVICES if w.h[d].online], key=lambda d: w.h[d].smell())
    if best != c.loc and w.h[best].smell() > 3:
        return A["go_" + best]
    return A["rest"]


if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    dna = (Path(__file__).resolve().parents[1] / "c1.dna").read_text()
    for ws in [7, 8, 9]:
        w, c = World(ws), Creature("oracle", dna)
        c.override = oracle
        ef = []
        for _ in range(days * TPD):
            w.step()
            c.step(w)
            ef.append(c.E / c.Emax())
        print(f"世界 {ws}: 孢子 {c.spore_count} 次，平均能量 {np.mean(ef):.2f}，最低 {min(ef):.2f}，"
              f"最终杂乱 {w.total_mess():.0f}，误吞 {w.log['compost_valuable']} 次")
