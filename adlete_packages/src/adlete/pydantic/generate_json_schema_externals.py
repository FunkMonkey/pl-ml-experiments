from typing import Any, Dict, List

from pydantic.json_schema import (
    GenerateJsonSchema,
    JsonRef,
    JsonSchemaMode,
    JsonSchemaValue,
    _DefinitionsRemapping,
)
from pydantic_core.core_schema import CoreSchema


def create_generate_json_schema_with_externals(
    defs_to_remove: List[str], refs_to_rename: Dict[str, str]
) -> type[GenerateJsonSchema]:
    """
    Creates a custom json schema generator, which excludes specific definitions and allows the
    altering of refs to them.

    This is a factory function creating a new class, so that we can pass information into it,
    since pydantic is itself instantiating the passed GenerateJsonSchema class.
    """

    class GenerateJsonSchemaWithExternals(GenerateJsonSchema):

        def generate_inner(self, schema: Any) -> JsonSchemaValue:
            # Skipping the creation of the definitions we want to remove.
            # Since generate_inner is called recursively for the whole tree, this will also
            # skip the processing of all child properties and thus other definitions.
            if "model_name" in schema and schema["model_name"] in defs_to_remove:
                return {}

            return super(GenerateJsonSchemaWithExternals, self).generate_inner(schema)

        def generate(
            self, schema: CoreSchema, mode: JsonSchemaMode = "validation"
        ) -> JsonSchemaValue:
            result = super(GenerateJsonSchemaWithExternals, self).generate(schema, mode)

            # Since generate_inner returns {}, it will still create stub definitions with a title.
            # We'll just remove them here.
            if "$defs" in result:
                defs = result["$defs"]
                for def_to_remove in defs_to_remove:
                    if def_to_remove in defs:
                        del defs[def_to_remove]

            return result

        def _build_definitions_remapping(self) -> _DefinitionsRemapping:

            remapping = super(GenerateJsonSchemaWithExternals, self)._build_definitions_remapping()
            json_remapping = remapping.json_remapping

            # Remap the refs, so they can point to external files.
            for key, ref in json_remapping.items():
                if ref in refs_to_rename:
                    json_remapping[key] = JsonRef(refs_to_rename[ref])

            return remapping

    return GenerateJsonSchemaWithExternals
