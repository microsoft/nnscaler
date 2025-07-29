from typing import List, Any
import os


def fname(path: str) -> str:
    return os.path.splitext(os.path.split(path)[-1])[0]


def unique(l: List[Any]):
    if len(l) == 1:
        return l[0]
    assert len(set(l)) == 1, f"{set(l)}"
    return l[0]


def select_column(grid: List[List[Any]], col: int) -> List[Any]:
    return [row[col] for row in grid]


def print_dict(d):
    for k, v in d.items():
        print(f"{k} {v}")
    print()


def print_list(l):
    for x in l:
        print(x)
    print()


def idempotent_update(d: dict, kws: dict):
    for key, value in kws.items():
        existing = d.get(key, None)
        if existing is not None:
            assert (
                existing == value
            ), f"Conflict key: {key}. \nExisting value: {d[key]}. \nIncoming value: {value}."
        else:
            d[key] = value
