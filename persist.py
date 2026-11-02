"""存档：把世界和生命体的状态写进一个 JSON 文件，再原样读回来。

巢穴（树莓派）重启不能丢命，所以要求读回来后逐 tick 和不中断时完全一致。
为此编码时保留类型：numpy 数组记下 dtype 和形状，numpy 标量和 Python 标量分开记，
元组和列表分开记。浮点数用 JSON 的最短往返表示，读回来逐位相同。
遇到不认识的类型直接报错，不会悄悄存错。

写文件先写临时文件再改名，断电时旧存档不会被写坏一半。
"""
import json
import os
import numpy as np

FORMAT = "c1-save"
VERSION = 1


def to_tree(x):
    """把状态编码成只含 JSON 原生类型的树。"""
    # numpy 类型先判断：np.float64 是 float 的子类，np.str_ 是 str 的子类
    if isinstance(x, np.ndarray):
        return {"__nd__": x.dtype.str, "shape": list(x.shape), "data": x.ravel().tolist()}
    if isinstance(x, np.generic):
        return {"__np__": x.dtype.str, "v": x.item()}
    if x is None or type(x) in (bool, int, float, str):
        return x
    if isinstance(x, tuple):
        return {"__tuple__": [to_tree(v) for v in x]}
    if isinstance(x, list):
        return [to_tree(v) for v in x]
    if isinstance(x, dict):
        assert all(isinstance(k, str) for k in x), "存档里的字典键只能是字符串"
        assert not any(k.startswith("__") for k in x), "以 __ 开头的键留给类型标记"
        return {k: to_tree(v) for k, v in x.items()}
    raise TypeError(f"存档不认识这种类型：{type(x)!r}")


def from_tree(x):
    if isinstance(x, list):
        return [from_tree(v) for v in x]
    if isinstance(x, dict):
        if "__nd__" in x:
            return np.array(x["data"], dtype=np.dtype(x["__nd__"])).reshape(x["shape"])
        if "__np__" in x:
            return np.dtype(x["__np__"]).type(x["v"])
        if "__tuple__" in x:
            return tuple(from_tree(v) for v in x["__tuple__"])
        return {k: from_tree(v) for k, v in x.items()}
    return x


def snapshot(world, creature=None):
    return dict(format=FORMAT, version=VERSION,
                world=world.state_dict(),
                creature=creature.state_dict() if creature is not None else None)


def dumps(world, creature=None):
    return json.dumps(to_tree(snapshot(world, creature)), ensure_ascii=False, allow_nan=True)


def loads(text):
    """读回 (world, creature)。creature 可能是 None（空白世界的存档）。"""
    from world import World
    from creature import Creature
    sd = from_tree(json.loads(text))
    assert sd.get("format") == FORMAT, "不是 C1 的存档"
    assert sd.get("version") == VERSION, f"存档版本 {sd.get('version')}，当前代码只认 {VERSION}"
    w = World()
    w.load_state_dict(sd["world"])
    c = Creature.from_state_dict(sd["creature"]) if sd["creature"] is not None else None
    return w, c


def save(path, world, creature=None):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(dumps(world, creature))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load(path):
    with open(path, encoding="utf-8") as f:
        return loads(f.read())
