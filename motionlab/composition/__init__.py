"""Phase 2 (ML-01.R3): Declarative Composition Graph, 9 Reusable Creative Recipes & Signal-Driven Solver."""

from motionlab.composition.planner import plan_composition_graph
from motionlab.composition.recipes import (
    RECIPE_CATALOG,
    RecipeSpec,
    build_recipe_capability_matrix,
    get_recipe,
    list_recipes,
)
from motionlab.composition.schema import (
    COMPOSITION_SCHEMA_VERSION,
    CompositionGraph,
    CompositionLayer,
    CompositionScene,
    validate_composition_graph,
)

__all__ = [
    "COMPOSITION_SCHEMA_VERSION",
    "CompositionGraph",
    "CompositionLayer",
    "CompositionScene",
    "RECIPE_CATALOG",
    "RecipeSpec",
    "build_recipe_capability_matrix",
    "get_recipe",
    "list_recipes",
    "plan_composition_graph",
    "validate_composition_graph",
]
