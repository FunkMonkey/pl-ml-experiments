import sys
from typing import Dict, List

from adlete.utils.io import write_json
from pydantic import BaseModel

from ..utils.paths import change_extension_to
from .generate_json_schema_externals import create_generate_json_schema_with_externals


def create_json_schema_file_for_model(
    model: type[BaseModel], defs_to_remove: List[str], refs_to_rename: Dict[str, str]
):
    """
    Creates a json-schema file for a pydantic model.
    Will infer the path based on the model's file.

    Pydantic has problems creating multiple json-schema files. If you want to split
    your models, you must call this function with different models and then clean
    the resulting json schemas using the parameters `defs_to_remove` and `refs_to_rename`.

    Args:
        model (type[BaseModel]): The pydantic model to create a json-schema from
        defs_to_remove (List[str]): JSON schema definitions to remove.
        refs_to_rename (Dict[str, str]): JSON schema refs to map.
    """
    class_path = sys.modules[model.__module__].__file__
    if class_path is None:
        raise Exception("Cannot infer filepath of given model!")

    schema_filename = change_extension_to(class_path, ".json")

    GenerateJsonSchemaWithExternals = create_generate_json_schema_with_externals(
        defs_to_remove, refs_to_rename
    )

    json_schema = model.model_json_schema(schema_generator=GenerateJsonSchemaWithExternals)
    write_json(schema_filename, json_schema)
