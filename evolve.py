"""以 C1-0 为始祖演化。

用法: python evolve.py [输出文件名，默认 evolved.dna]
注意：c1.dna 是 C1 本人，脚本拒绝覆盖它。

每一代所有个体在同一个新世界里各活 TRIAL_DAYS 天（世界每代都换，防止只适应一个世界）。
适应度只看"活着"：非孢子时间占比为主，能量和伤势为辅。
不看它整理了多少东西：对用户有用，只能通过"吃杂乱才能活"间接体现。
"""
import json
import os
import sys
import time
import numpy as np
from multiprocessing import Pool
from genome import random_dna, mutate, crossover, upgrade
from world import World, TPD
from creature import Creature

POP, GENS, TRIAL_DAYS = 32, 30, 12
ELITE, PARENTS = 4, 10


def fitness(args):
    dna, wseed = args
    w, c = World(wseed), Creature("x", dna)
    act, Ef, dmg = 0, 0.0, 0.0
    T = TRIAL_DAYS * TPD
    for t in range(T):
        w.step()
        c.step(w)
        act += 0 if c.spore else 1
        Ef += c.E / c.Emax()
        dmg += c.damage
    return act / T + 0.2 * Ef / T - 0.3 * dmg / T, act / T


def main():
    # 始祖以 c1_0.dna 为准；基因表追加过新基因时，补成当前版本（新基因取中性值）
    if os.path.exists("c1_0.dna"):
        founder = upgrade(open("c1_0.dna").read().strip())
    else:
        founder = random_dna(1)
        open("c1_0.dna", "w").write(founder)
    out = sys.argv[1] if len(sys.argv) > 1 else "evolved.dna"
    assert out != "c1.dna", "c1.dna 是 C1 本人的基因组，不要覆盖；新个体请另起文件名"
    pop = [founder] + [mutate(founder, 0.08, 100 + i) for i in range(POP - 1)]
    hist, t0 = [], time.time()
    with Pool() as pool:
        for g in range(GENS):
            res = pool.map(fitness, [(d, 1000 + g) for d in pop])
            fit = np.array([r[0] for r in res])
            act = np.array([r[1] for r in res])
            order = np.argsort(-fit)
            hist.append(dict(gen=g, best=float(fit[order[0]]), mean=float(fit.mean()),
                             best_active=float(act[order[0]]), mean_active=float(act.mean()),
                             founder_fit=float(fit[0]) if g == 0 else None))
            print(f"第{g:2d}代  最佳适应度 {fit[order[0]]:.3f}  平均 {fit.mean():.3f}  "
                  f"最佳活跃时间 {act[order[0]]:.2f}  平均活跃 {act.mean():.2f}  ({time.time() - t0:.0f}s)", flush=True)
            parents = [pop[i] for i in order[:PARENTS]]
            nxt = [pop[i] for i in order[:ELITE]]
            k = 0
            while len(nxt) < POP:
                k += 1
                rs = np.random.default_rng(g * 1000 + k)
                a, b = rs.choice(PARENTS, 2, replace=False)
                child = crossover(parents[a], parents[b], g * 1000 + k)
                nxt.append(mutate(child, 0.01, g * 7919 + k))
            pop = nxt
        # 最后一代：在 3 个新世界里复评精英，选最稳的当 C1
        cands = pop[:ELITE]
        scores = [np.mean([r[0] for r in pool.map(fitness, [(d, s) for s in (5001, 5002, 5003)])]) for d in cands]
    best = cands[int(np.argmax(scores))]
    open(out, "w").write(best)
    json.dump(hist, open(out.replace(".dna", "") + "_evolution.json", "w"), indent=1)
    print(f"冠军写入 {out}，复评适应度", round(max(scores), 3))


if __name__ == "__main__":
    main()
