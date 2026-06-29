from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TrainConfigValidator(BaseModel):
    model_name: str = Field(..., description="Registered model name")
    batch_size: int = Field(gt=0, description="Batch size per GPU")
    max_epochs: int = Field(gt=0, description="Maximum training epochs")
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    grad_accumulation_steps: int = Field(default=1, ge=1)
    image_size: tuple[int, int] | None = None

    @model_validator(mode="after")
    def check_grad_accumulation(self) -> "TrainConfigValidator":
        if self.grad_accumulation_steps > 1 and self.precision == "fp32":
            self.precision = "fp16"
        return self

    @model_validator(mode="after")
    def check_image_size_even(self) -> "TrainConfigValidator":
        if self.image_size is not None:
            h, w = self.image_size
            if h % 32 != 0:
                raise ValueError(f"image_size height ({h}) should be divisible by 32 for most backbones.")
            if w % 32 != 0:
                raise ValueError(f"image_size width ({w}) should be divisible by 32 for most backbones.")
        return self


def validate_train_config(config: dict) -> list[str]:
    warnings: list[str] = []
    try:
        TrainConfigValidator(**config)
    except Exception as e:
        warnings.append(f"[ERROR] Config validation failed: {e}")
    batch_size = config.get("batch_size", 1)
    max_epochs = config.get("max_epochs", 1)
    if batch_size * max_epochs < 100:
        warnings.append("[WARN] batch_size * max_epochs is very small; training may not converge.")
    return warnings
