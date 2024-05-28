# -------------------------------------------------------------------------##
# You can run this python script directly!                                 ##
#                                                                          ##
# Generate the JSON schemas for this package.                              ##
# -------------------------------------------------------------------------##


from adlete.pydantic.generate_json_schema_file import create_json_schema_file_for_model
from adlete.torch.lightning_utils.config_schema import LightningConfigRoot
from adlete.training.config_schema import TrainingConfigRoot

create_json_schema_file_for_model(LightningConfigRoot, [], {})

training_defs_to_remove = ["TrainerConfig"]
training_refs_to_rename = {
    "#/$defs/TrainerConfig": "../torch/lightning_utils/config_schema.json#/$defs/TrainerConfig"
}

create_json_schema_file_for_model(
    TrainingConfigRoot, training_defs_to_remove, training_refs_to_rename
)
