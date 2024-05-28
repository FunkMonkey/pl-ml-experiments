from typing import Any


def get_nested_prop(obj: object, prop_path: str) -> Any:
    fields = prop_path.split(".")
    curr_prop = obj

    for field in fields:
        if hasattr(curr_prop, field):
            curr_prop = getattr(curr_prop, field)
        else:
            return None

    return curr_prop


def set_nested_prop(obj: object, prop_path: str, value: Any) -> bool:
    fields = prop_path.split(".")
    curr_prop = obj

    for field in fields[:-1]:
        if hasattr(curr_prop, field):
            curr_prop = getattr(curr_prop, field)
        else:
            return False

    setattr(curr_prop, fields[-1], value)
    return True
