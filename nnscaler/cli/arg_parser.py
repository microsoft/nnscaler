#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import os
import copy

from typing import List, Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass, field, is_dataclass, asdict
import enum
import ast
import regex


try:
    from types import UnionType
except ImportError:
    UnionType = None  # for python < 3.10


_TYPE_KEY = '__type'
_VALUE_TYPE_KEY = '__value_type'
_VALUE_KEY = 'value'

# Keys for metadata in dataclass fields
# These keys are used to control the deserialization and normalization behavior

# specify a custom deserialization function,
# the return value of this function will be assigned to the dataclass field directly.
# So it should return a value of the type specified in the dataclass field
DESERIALIZE_KEY = 'deserialize'
# specify a custom normalization function.
# The value returned by this function will be further deserialized with default deserialization logic.
NORMALIZE_KEY = 'normalize'
# if set to True, the field will be skipped during deserialization
# You can use `__post_init__` to handle the deserialization of the field.
SKIP_DESERIALIZATION_KEY = 'skip_deserialization'


class _KeyNotFoundError(KeyError):
    pass


def parse_args(argv: List[str]) -> dict:
    raw_args = {}
    last_key = None
    for v in argv:
        if isinstance(v, str) and v.startswith('--'):
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


def merge_args(args: dict, argv: List[str]):
    """
    Please note that this function will modify the args in place.
    """
    _merge_args(args, parse_args(argv))


def _merge_args(args: dict, new_args: dict):
    """
    Note: values in new_args can only be dict or str or None.
    """
    def _is_removed_key(k):
        return isinstance(k, str) and k.endswith('!')

    for k, v in new_args.items():
        if _is_removed_key(k):
            # if the key ends with '!', we will remove the key from args
            args.pop(k, None) # a little trick to support merge self
            k = k[:-1]
            args.pop(k, None)
            continue
        if k not in args or not isinstance(args[k], (dict, list)):
            # if the existing value is not a dict/list, we will overwrite it with the new value
            # for example, if args is {'a': 1} and new_args is {'a': {'b': 2}},
            # we will overwrite args['a'] with new_args['a']
            if isinstance(v, dict):
                new_v = copy.deepcopy(v)
                # merge self trick is here.
                # directly assign v to args[k] will not work
                # because v can have removed keys.
                _merge_args(new_v, v)
                args[k] = new_v # do we need to keep the empty dict?
            else:
                args[k] = v
        elif isinstance(args[k], dict):
            if isinstance(v, dict):
                _merge_args(args[k], v)
            else:
                args[k] = v
        else:
            assert isinstance(args[k], list)
            # we only update per-element value if the new value is a dict
            if isinstance(v, dict) \
                and all(
                    isinstance(item, int) or
                    (isinstance(item, str) and item.isdigit()) or
                    (_is_removed_key(item) and item[:-1].isdigit())
                    for item in v.keys()
                ):
                current_value = {str(idx): item for idx, item in enumerate(args[k])}
                new_value = {str(idx): item for idx, item in v.items()}
                _merge_args(current_value, new_value)
                current_value = {int(k): v for k, v in current_value.items()}
                args[k] = [None] * (max(current_value.keys()) + 1)
                for nk, nv in current_value.items():
                    args[k][nk] = nv
            else:
                args[k] = v


def resolve_args(args: dict):
    """
    Substitute the args with the value from the args.
    For example, if args is {'a': '$(b)', 'b': 'c'}, then
    it will be updated to {'a': 'c', 'b': 'c'}.
    """
    pattern = r'(\$\{[^}]+\}|\$\([^)]+\))'

    def _is_variable(var_path):
        return isinstance(var_path, str) and (
            (var_path.startswith('$(') and var_path.endswith(')')) or
            (var_path.startswith('${') and var_path.endswith('}'))
        )

    def _get_variable(var_path: Any) -> Optional[str]:
        if not _is_variable(var_path):
            return None
        return var_path[2:-1]

    def _get_variables(var_path: str) -> List[str]:
        """
        Get all variables in the var_path.
        For example, if var_path is 'a$(a.b.c)b$(c.d)c', it will return ['a.b.c', 'c.d'].
        """
        # use regex to find all variables in the var_path
        matches = regex.findall(pattern, var_path)
        return [_get_variable(m) for m in matches]

    def _resolve_variables(var_path: Any, resolved_vars: dict[str, str]) -> str | Any:
        """
        Resolve all variables in the var_path by replacing them with their values.
        For example, if var_path is 'a$(b.c)d$(e.f)g', and resolved_vars is {'b.c': 'x', 'e.f': 'y'},
        it will return 'axdyg'.
        """
        # special case, this will keep the type of the variable
        if _is_variable(var_path):
            return resolved_vars[_get_variable(var_path)]

        # always return a string
        var_path = regex.sub(
            pattern,
            lambda m: str(resolved_vars[_get_variable(m.group(0))]),
            var_path
        )
        var_path = var_path.replace(r'$\(', '$(').replace(r'$\{', '${')  # escape the variable syntax
        return var_path

    def _get_value(data, var_path: list[Any]):
        for key in var_path:
            if isinstance(data, list):
                data = data[int(key)]
            elif key in data:
                data = data[key]
            else:
                raise _KeyNotFoundError(f"{var_path} not found in args")
        return data

    def _set_value(data, var_path: list[Any], value):
        value = copy.deepcopy(value)
        for key in var_path[:-1]:
            if isinstance(data, list):
                data = data[int(key)]
            elif key in data:
                data = data[key]
            else:
                raise _KeyNotFoundError(f"{var_path} not found in args")

        if isinstance(data, list):
            data[int(var_path[-1])] = value
        else:
            data[var_path[-1]] = value

    pending_values = set()
    def _resolve(var_path: list[Any], value: Any):
        if isinstance(value, dict):
            for k, v in value.items():
                _resolve(var_path + [k], v)
            return value
        elif isinstance(value, list):
            for i, v in enumerate(value):
                _resolve(var_path + [i], v)
            return value
        elif isinstance(value, str):
            ref_keys = _get_variables(value)
            ref_values = {}
            for ref_key in ref_keys:
                if ref_key in pending_values:
                    raise ValueError(f"Circular reference detected for {ref_key}")
                pending_values.add(ref_key)
                ref_var_path = ref_key.split('.')
                try:
                    raw_ref_value = _get_value(args, ref_var_path)
                    ref_values[ref_key] = _resolve(ref_var_path, raw_ref_value)
                except _KeyNotFoundError as e:
                    if ref_key in os.environ:
                        ref_values[ref_key] = os.environ[ref_key]
                    else:
                        raise
                pending_values.remove(ref_key)
            resolved_value = _resolve_variables(value, ref_values)
            _set_value(args, var_path, resolved_value)
            return resolved_value
        else:
            return value
    _resolve([], args)


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
    metadata: dict = field(default_factory=dict)


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
    type_dict: dict[str, _TypeInfo] = {}
    for k, v in dataclass_type.__dataclass_fields__.items():
        if v.metadata.get(SKIP_DESERIALIZATION_KEY, False) or DESERIALIZE_KEY in v.metadata:
            # if the field is marked as skip_deserialization,
            # or if it has a custom deserialize function,
            # we don't need to extract the type information
            type_dict[k] = _TypeInfo(type=None)
        else:
            type_dict[k] = _get_type_info_from_annotation(v.type)
        type_dict[k].metadata = v.metadata
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
        # special handling for 'false'/'true'.
        # 'False'/'True' are handled in ast.literal_eval
        if value == 'false':
            return False
        if value == 'true':
            return True
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

        if deserialize_func := ti.metadata.get(DESERIALIZE_KEY, None):
            v = deserialize_func(v)
            member_values[key] = v
            continue

        if normalize_func := ti.metadata.get(NORMALIZE_KEY, None):
            v = normalize_func(v)
            # will continue to process the value

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
