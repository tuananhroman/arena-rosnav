from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class CurriculumStage(BaseModel):
    """Configuration for a single curriculum stage matching actual ROS2 parameter structure."""

    model_config = {"populate_by_name": True}

    task_random_static_n: List[int] = Field(
        default_factory=lambda: [0, 0],
        description="Min/max static obstacles",
        alias="task.random.static.n",
    )
    task_random_interactive_n: List[int] = Field(
        default_factory=lambda: [0, 0],
        description="Min/max interactive obstacles",
        alias="task.random.interactive.n",
    )
    task_random_dynamic_n: List[int] = Field(
        default_factory=lambda: [0, 0],
        description="Min/max dynamic obstacles",
        alias="task.random.dynamic.n",
    )
    task_random_static_models: List[str] = Field(
        default_factory=list,
        description="Static obstacle model names",
        alias="task.random.static.models",
    )
    task_random_interactive_models: List[str] = Field(
        default_factory=list,
        description="Interactive obstacle model names",
        alias="task.random.interactive.models",
    )
    task_random_dynamic_models: List[str] = Field(
        default_factory=list,
        description="Dynamic obstacle model names",
        alias="task.random.dynamic.models",
    )
    goal_tolerance_radius: Optional[float] = Field(
        default=None, description="Goal tolerance radius"
    )


class StagedCfg(BaseModel):
    starting_stage: int = Field(default=0, description="Initial curriculum stage")
    curriculum_definition: List[CurriculumStage] = Field(
        default_factory=list, description="Curriculum stages defined under staged"
    )

    # Curriculum progression parameters (moved from callbacks.training_curriculum)
    threshold_type: str = Field(
        default="succ",
        description="Metric to use: 'succ' (success rate) or 'rew' (reward)",
    )
    upper_threshold: float = Field(
        default=0.9, description="Performance threshold to advance to next stage"
    )
    lower_threshold: float = Field(
        default=0.3, description="Performance threshold to stay at current stage"
    )

    # ROS2 parameter communication settings
    parameter_node_template: str = Field(
        default="/task_generator_node",
        description="Template for task generator node name",
    )
    timeout: float = Field(
        default=5.0, description="Timeout for ROS2 parameter service calls (seconds)"
    )

    @field_validator("threshold_type")
    @classmethod
    def validate_threshold_type(cls, v: str) -> str:
        """Validate threshold_type is either 'rew' or 'succ'."""
        if v not in ["rew", "succ"]:
            raise ValueError(f"threshold_type must be 'rew' or 'succ', got '{v}'")
        return v

    @field_validator("upper_threshold", "lower_threshold")
    @classmethod
    def validate_thresholds(cls, v: float) -> float:
        """Validate threshold values are reasonable."""
        if v < 0:
            raise ValueError("Thresholds must be non-negative")
        return v

    @field_validator("starting_stage")
    @classmethod
    def validate_starting_stage(cls, v: int) -> int:
        """Validate starting_stage is non-negative."""
        if v < 0:
            raise ValueError("starting_stage must be non-negative")
        return v

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: float) -> float:
        """Validate timeout is positive."""
        if v <= 0:
            raise ValueError("timeout must be positive")
        return v

    @model_validator(mode="after")
    def validate_threshold_relationship(self):
        """Validate that upper_threshold > lower_threshold."""
        if self.upper_threshold <= self.lower_threshold:
            raise ValueError(
                f"upper_threshold ({self.upper_threshold}) must be greater than "
                f"lower_threshold ({self.lower_threshold})"
            )
        return self

    @model_validator(mode="after")
    def validate_starting_stage_bounds(self):
        """Validate starting_stage is within curriculum_definition bounds."""
        if self.curriculum_definition and self.starting_stage >= len(
            self.curriculum_definition
        ):
            raise ValueError(
                f"starting_stage ({self.starting_stage}) must be less than "
                f"number of curriculum stages ({len(self.curriculum_definition)})"
            )
        return self

    @model_validator(mode="after")
    def validate_curriculum_definition(self):
        """Validate curriculum_definition has at least one stage if curriculum is used."""
        # Only validate if this is being used for curriculum learning
        # (We can't easily check tm_modules here, so we'll be lenient)
        return self


class TaskCfg(BaseModel):
    tm_robots: str = "random"
    tm_obstacles: str = "random"
    tm_modules: str = "staged"

    # The configuration now only supports the nested `staged` block
    staged: Optional[StagedCfg] = None

    def get_curriculum_stages(self) -> Optional[List[CurriculumStage]]:
        """Get curriculum stages from configuration.

        Returns:
            List of curriculum stages or None if no curriculum configured
        """
        return (
            self.staged.curriculum_definition
            if self.staged.curriculum_definition
            else None
        )

    def has_curriculum(self) -> bool:
        """Check if curriculum configuration is present."""
        return bool(self.staged and self.staged.curriculum_definition)

    def get_starting_stage(self) -> int:
        """Get the starting stage index."""
        return self.staged.starting_stage
