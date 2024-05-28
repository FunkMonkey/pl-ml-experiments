# -------------------------------------------------------------------------##
# You can run this python script directly!                                 ##
#                                                                          ##
# Generate the JSON schemas for this package.                              ##
# -------------------------------------------------------------------------##

from adlete.pydantic.generate_json_schema_file import create_json_schema_file_for_model

from experiments.knowledge_tracing.datasets.config_schema import KTDatasetConfigRoot
from experiments.knowledge_tracing.torch.dkt.config_schema import DKTConfigFile

create_json_schema_file_for_model(KTDatasetConfigRoot, [], {})


training_defs_to_remove = ["KTDatasetConfig", "TrainingConfig"]
training_refs_to_rename = {
    "#/$defs/KTDatasetConfig": "../../datasets/config_schema.json#/$defs/KTDatasetConfig",
    "#/$defs/TrainingConfig": (
        "../../../../../adlete_packages/src/adlete/training/config_schema.json"
        "#/$defs/TrainingConfig"
    ),
}
create_json_schema_file_for_model(DKTConfigFile, training_defs_to_remove, training_refs_to_rename)
