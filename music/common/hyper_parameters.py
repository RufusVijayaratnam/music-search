from pydantic import BaseModel, ConfigDict
import torch


class HyperParameters(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    device: torch.device
    dtype: torch.dtype
