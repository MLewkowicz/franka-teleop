"""LLM-synthesized VoxPoser value maps from world-frame scene boxes.

    task + [SceneBox]
        -> PlannerLMP        -> [{affordance, avoidance, arrival_radius, label}]
        -> MapLMP x2         -> affordance / avoidance voxel grids
        -> voxposer.ValueMap -> smoothed, gradients precomputed
        -> save_stages(.npz) -> loaded by PositionFieldSteering at deploy time

LangSteer must be importable (`deploy_diffuser_actor._wire_langsteer`, or the
sys.path insert the CLI scripts do) before anything here is called; the
`voxposer` imports are all deferred to call time for that reason.
"""

from clear_franka.value_map_llm.artifact import load_stages, save_stages
from clear_franka.value_map_llm.interface import (
    BoxSceneInterface,
    ObjectResolutionError,
)
from clear_franka.value_map_llm.llm import LLMBackend
from clear_franka.value_map_llm.synthesize import (
    SynthesizedStage,
    synthesize_and_save,
    synthesize_value_maps,
    workspace_bounds_from_boxes,
)

__all__ = [
    "BoxSceneInterface",
    "LLMBackend",
    "ObjectResolutionError",
    "SynthesizedStage",
    "load_stages",
    "save_stages",
    "synthesize_and_save",
    "synthesize_value_maps",
    "workspace_bounds_from_boxes",
]
