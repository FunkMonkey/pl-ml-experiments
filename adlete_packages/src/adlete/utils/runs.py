import datetime
import random
from typing import List


def generate_run_id():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def generate_run_id_with_name(
    names: List[str], prefix: str = "", randomizer: random.Random | None = None
):
    name = random.choice(names) if randomizer is None else randomizer.choice(names)
    appendix = f" {name}" if prefix == "" else f" {prefix} {name}"
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + appendix


global_run_id = generate_run_id()


def set_global_run_id(run_id: str):
    global global_run_id
    global_run_id = run_id


def get_global_run_id():
    return global_run_id
