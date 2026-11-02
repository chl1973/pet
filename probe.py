"""快速体检：同一个 C1 基因组在几个不同世界里各活 30 天，看是否稳定。"""
import sys
import numpy as np
from genome import ACTIONS
from world import World, TPD
from creature import Creature

dna = open("c1.dna").read()
days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
seeds = [7, 8, 9]
for learn in [True, False]:
    for ws in seeds:
        w, c = World(ws), Creature("C1", dna, learn=learn)
        acts, Ef, sp = [], [], []
        for t in range(days * TPD):
            w.step(); c.step(w)
            acts.append(c.action); Ef.append(c.E / c.Emax()); sp.append(c.spore)
        acts, Ef, sp = np.array(acts), np.array(Ef), np.array(sp)
        line = []
        for d0 in range(0, days, 10):
            s = slice(d0 * TPD, (d0 + 10) * TPD)
            aa = acts[s][acts[s] >= 0]
            pd = np.bincount(aa, minlength=8) / max(1, len(aa))
            ent = -(pd[pd > 0] * np.log(pd[pd > 0])).sum() / np.log(8)
            line.append(f"d{d0}: E{Ef[s].mean():.2f} 孢{sp[s].mean():.2f} 熵{ent:.2f} top={ACTIONS[pd.argmax()]}{pd.max():.2f}")
        print(("学" if learn else "无"), ws, f"孢子{c.spore_count:3d} 体型{c.size:.2f} 混乱{w.total_mess():6.0f} |", " | ".join(line))
