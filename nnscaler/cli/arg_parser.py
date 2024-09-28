#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import List, Tuple, Dict, Any, Union
from dataclasses import dataclass, is_dataclass, asdict
import enum
import ast


try:
    from types import UnionType
except ImportError:
    UnionType = None  # for python < 3.10


_TYPE_KEY = '__type'
_VALUE_TYPE_KEY = '__value_type'
_VALUE_KEY = 'value'


def parse_args(argv: List[str]) -> dict:
    raw_args = {}
    last_key = None
    for v in argv:
        if v.startswith('--'):
            if '=' in v:
                k, v = v[2:].split('=', 1)
                raw_args[k] = v
                last_key = None
            else:
                k = v[2:]
                raw_args[k] = None
                last_key = k
        else:
            if not last_key:
                raise ValueError(f"invalid argument {v}")
            raw_args[last_key] = v
            last_key = None

    args = {}
    for k, v in raw_args.items():
        k = k.replace('-', '_')
        keys = k.split('.')
        current = args
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = v

    return args


def merge_args(args: dict, new_args: dict):
    for k, v in new_args.items():
        if k in args and isinstance(args[k], dict) and isinstance(v, dict):
            merge_args(args[k], v)
        else:
            args[k] = v


def _fix_any(type_):
    return None if type_ == Any else type_


def _fix_optional(type_info):
    if getattr(type_info, '__origin__', None) == Union \
        or (UnionType and isinstance(type_info, UnionType)):
        args = getattr(type_info, '__args__', None)
        if len(args) != 2 or (args[1] != type(None) and args[0] != type(None)):
            # when multiple types are allowed,
            # we don't do any conversion
            # let's the user to handle it
            return Any
        if args[1] == type(None):
            return _fix_optional(args[0])
        else:
            return args[1]
    return type_info


def _fix_type(type_info, raise_on_nested=True):
    type_info = _fix_optional(type_info)
    type_info = _fix_any(type_info)
    if raise_on_nested and getattr(type_info, '__args__', None):
        raise ValueError(f"Nested type {type_info} is not allowed here.")
    return type_info


@dataclass
class _TypeInfo:
    type: Any = None
    key_type: Any = None
    value_type: Any = None
    item_type: Any = None


def _get_type_info_from_annotation(type_info):
    type_info = _fix_type(type_info, False)
    if type_info is None or type_info == Any:
        return _TypeInfo(type=None)
    if type_info in (list, List):
        return _TypeInfo(type=list)
    if type_info in (dict, Dict):
        return _TypeInfo(type=dict)
    if type_info in (tuple, Tuple):
        return _TypeInfo(type=tuple)

    origin = getattr(type_info, '__origin__', None)
    args = getattr(type_info, '__args__', None)

    if origin in (List, list):
        if len(args) != 1:
            raise ValueError(f"Invalid list type {type_info}")
        return _TypeInfo(type=list, item_type=_fix_type(args[0]))
    elif origin in (Dict, dict):
        if len(args) != 2:
            raise ValueError(f"Invalid dict type {type_info}")
        return _TypeInfo(type=dict, key_type=_fix_type(args[0]), value_type=_fix_type(args[1]))
    elif origin in (Tuple, tuple):
        if len(args) != 2 or args[1] != Ellipsis:
            raise ValueError(f"Invalid tuple type {type_info}")
        return _TypeInfo(type=tuple, item_type=_fix_type(args[0]))
    else:
        if type_info.__module__ == 'typing':
            raise ValueError(f"Unsupported type {type_info}")
        return _TypeInfo(type=type_info)


def _get_type_info(dataclass_type) -> Dict[str, _TypeInfo]:
    if not is_dataclass(dataclass_type):
        raise ValueError(f"{dataclass_type} is not a dataclass")
    type_dict = {}
    for k, v in dataclass_type.__dataclass_fields__.items():
        type_dict[k] = _get_type_info_from_annotation(v.type)
    return type_dict


def _is_primitive_type(data_type):
    """
    We only support int, str, bool, float as primitive types.
    """
    return data_type in (int, str, bool, float)


def _guess_deserialize_object(value):
    if isinstance(value, dict):
        if _VALUE_KEY in value and _VALUE_TYPE_KEY in value and len(value) == 2:
            # keep as it is if it is a value object
            return value
        return {_guess_deserialize_object(k): _guess_deserialize_object(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_guess_deserialize_object(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_guess_deserialize_object(v) for v in value)
    if isinstance(value, str):
        try:
            # try to parse as literal
            # if failed, return as it is
            # Please note that if there is no type annotation,
            # you should provide the value in python code format
            # for example `[a, b, c]` will return string as a whole
            # but `['a', 'b', 'c']` will return a list of strings
            return ast.literal_eval(value)
        except Exception:
            return value
    return value


def _deserialize_object(value, value_type):
    """
    deserialize object based on single value type:
    1. If no value type or collective type, try to guess its value type.
    2. if it is primitive types, return value_type(value)
    3. If it is a dataclass, ask dataclass to deserialize.
    4. Otherwise, we will return as it is
    """
    if not value_type or value_type in (dict, list, tuple):
        return _guess_deserialize_object(value)
    try:
        if value is None:
            return value
        if isinstance(value, value_type):
            return value
        if issubclass(value_type, enum.Enum):
            try:
                return value_type[value]  # first treat as enum name
            except KeyError:
                return value_type(value)  # then treat as enum value
        if value_type == bool and isinstance(value, str):
            if value.lower() in ('true', '1'):
                return True
            elif value.lower() in ('false', '0'):
                return False
            else:
                raise ValueError(f"Failed to deserialize {value} to {value_type}")
        if _is_primitive_type(value_type):
            return value_type(value)
    except Exception as ex:
        raise ValueError(f"Failed to deserialize {value} to {value_type}") from ex

    if is_dataclass(value_type):
        return deserialize_dataclass(value, value_type)

    return value


def deserialize_dataclass(value, value_type):
    if not isinstance(value, dict):
        raise ValueError(f"Expecting dict, but got {value}")
    if not is_dataclass(value_type):
        raise ValueError(f"{value_type} is not a dataclass")

    type_info = _get_type_info(value_type)
    member_values = {}
    used_keys = set()
    for key, ti in type_info.items():
        if not key in value:
            continue
        used_keys.add(key)
        v = value[key]
        if ti.type is bool and v is None:
            v = True   # set bool to True if it shows up in cmd line
        if v is None:
            continue

        if ti.type in (list, tuple, dict, type(None)) and isinstance(v, str):
            v = ast.literal_eval(v)

        if isinstance(v, (list, tuple, dict)) and not ti.type:
            ti.type = type(v)

        if ti.item_type or ti.key_type or ti.value_type:
            if ti.type in (list, tuple):
                if isinstance(v, (list, tuple)):
                    v = ti.type(_deserialize_object(x, ti.item_type) for x in v)
                elif isinstance(v, dict):
                    v_dict = {_deserialize_object(k, int): _deserialize_object(v, ti.item_type) for k, v in v.items()}
                    v = [None] * (max(v_dict.keys()) + 1)
                    for k, x in v_dict.items():
                        v[k] = x
                    v = ti.type(v)
                else:
                    raise ValueError(f"Invalid value {v} for {value_type}")
            elif ti.type == dict:
                v = {_deserialize_object(k, ti.key_type): _deserialize_object(v, ti.value_type) for k, v in v.items()}
        else:
            v = _deserialize_object(v, ti.type)

        if v is not None: # for none values, use default value.
            member_values[key] = v
    if set(value.keys()) - used_keys:
        raise ValueError(f"Unknown members {set(value.keys()) - used_keys} for {value_type}")
    return value_type(**member_values)
