"""孵化 C1，让它在模拟世界里独立生活，并记录每个 tick 的状态。

用法: python run.py [天数] [世界种子]
"""
import json
import sys
import time
import numpy as np
from genome import random_dna, decode, NM
from world import World, TPD, DEVICES, HABITATS
from creature import Creature

LOC_ID = {n: i for i, n in enumerate(HABITATS)}
EVENT_ID = {"": 0, "evicted": 1, "arrived": 2, "depart": 3, "toxic": 4, "rough": 5,
            "task": 6, "spore": 7, "revive": 8}
ATE_ID = {None: 0, "temp": 1, "dup": 2, "misplaced": 3, "sorted": 4}


def simulate(dna, days, world_seed, learn=True, creature=True, name="C1"):
    w = World(world_seed)
    c = Creature(name, dna, learn=learn) if creature else None
    T = days * TPD
    R = {k: np.zeros(T) for k in [
        "E", "Emax", "damage", "toxin", "size", "hue", "speed", "gained", "basal",
        "arousal", "action_cost", "user_here", "cursor", "touch", "warmth", "crowd",
        "local_smell", "spore", "transit"]}
    R["loc"] = np.zeros(T, int)
    R["action"] = np.full(T, -1, int)
    R["event"] = np.zeros(T, int)
    R["ate"] = np.zeros(T, int)
    R["spat"] = np.zeros(T, int)
    R["m"] = np.zeros((T, NM))
    R["expr"] = np.zeros((T, 5))
    R["mess"] = np.zeros((T, len(DEVICES)))
    R["online"] = np.zeros((T, len(DEVICES)))
    R["user"] = np.zeros((T, len(DEVICES)))
    for t in range(T):
        w.step()
        if c:
            c.step(w)
            R["E"][t], R["Emax"][t] = c.E, c.Emax()
            R["damage"][t], R["toxin"][t], R["size"][t] = c.damage, c.toxin, c.size
            R["hue"][t], R["speed"][t] = c.phenotype()
            R["gained"][t] = c.spent["gained"]
            R["basal"][t], R["arousal"][t], R["action_cost"][t] = (
                c.spent["basal"], c.spent["arousal"], c.spent["action"])
            R["loc"][t] = LOC_ID[c.loc]
            R["transit"][t] = 1 if c.transit else 0
            R["spore"][t] = 1 if c.spore else 0
            R["action"][t] = c.action
            R["event"][t] = EVENT_ID[c.event]
            if c.ate:
                R["ate"][t] = ATE_ID[c.ate["kind"]] if not c.ate["spat"] else 0
                R["spat"][t] = 1 if c.ate["spat"] else 0
            R["m"][t] = c.m
            e = c.expression()
            R["expr"][t] = [1.0, e[7], e[12], e[15], e[18]]
            here = c.loc if not c.transit else "nest"
            R["user_here"][t] = 1 if here != "nest" and w.h[here].user else 0
            R["cursor"][t], R["touch"][t] = c.last_social["cursor_near"], c.last_social["touch"]
            _, R["warmth"][t], R["crowd"][t] = w.climate(here)
            R["local_smell"][t] = w.h[here].smell() if here != "nest" else 0
        for i, d in enumerate(DEVICES):
            R["mess"][t, i] = w.h[d].mess()
            R["online"][t, i] = w.h[d].online
            R["user"][t, i] = w.h[d].user
    return R, w, c


if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    wseed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    dna = open("c1.dna").read()          # 演化选出的 C1（evolve.py 写入）
    founder = open("c1_0.dna").read()    # 始祖 C1-0：种子 1 随机生成
    t0 = time.time()
    runs = {
        "C1": simulate(dna, days, wseed, learn=True),
        "C1_nolearn": simulate(dna, days, wseed, learn=False, name="C1-nolearn"),
        "C1_0": simulate(founder, days, wseed, learn=True, name="C1-0"),
        "baseline": simulate(dna, days, wseed, creature=False),
    }
    print(f"模拟 {days} 天 x {len(runs)} 组，用时 {time.time() - t0:.1f}s")
    for k, (R, w, c) in runs.items():
        np.savez_compressed(f"log_{k}.npz", **R)
        out = dict(world_log=w.log)
        if c:
            out.update(spores=c.spore_count, final_size=round(c.size, 3), epi=c.epi,
                       eaten={k2: int((R["ate"] == v).sum()) for k2, v in
                              [("temp", 1), ("dup", 2), ("misplaced_wrong", 3), ("sorted", 4)]},
                       spat=int(R["spat"].sum()),
                       mean_energy_frac=round(float((R["E"] / R["Emax"]).mean()), 3),
                       spore_frac=round(float(R["spore"].mean()), 3))
        out["mean_mess_last14d"] = round(float(R["mess"][-14 * TPD:].sum(1).mean()), 1)
        out["final_mess"] = round(float(R["mess"][-1].sum()), 1)
        print(k, json.dumps(out, ensure_ascii=False))
        json.dump(out, open(f"summary_{k}.json", "w"), ensure_ascii=False, indent=1)
