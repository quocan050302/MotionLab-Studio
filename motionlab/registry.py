"""Phase D1: Effect registry loader with schema, capability, license, and acyclic fallback verification."""

from __future__ import annotations

from typing import Any

from motionlab.effects.catalog import build_effect_catalog
from motionlab.schemas import EffectSpec, SchemaValidationError


class EffectRegistry:
    """Validated, cycle-checked, immutable registry of ML-01 effects."""

    def __init__(self, specs: list[EffectSpec]) -> None:
        self._by_id: dict[str, EffectSpec] = {}
        self._by_name: dict[str, EffectSpec] = {}
        for spec in specs:
            if spec.id in self._by_id:
                raise SchemaValidationError(f"Duplicate effect id: {spec.id}")
            if spec.name in self._by_name:
                raise SchemaValidationError(f"Duplicate effect name: {spec.name}")
            if spec.min_duration_s <= 0 or spec.max_duration_s < spec.min_duration_s:
                raise SchemaValidationError(f"Invalid duration bounds on {spec.id}")
            if spec.max_intensity <= 0 or spec.max_intensity > 1.0:
                raise SchemaValidationError(f"Invalid max_intensity on {spec.id}: {spec.max_intensity}")
            if spec.engine not in spec.supported_engines:
                raise SchemaValidationError(f"Primary engine {spec.engine} not in supported_engines on {spec.id}")
            # Validate default parameter values against own schema
            spec.validate_params({})
            self._by_id[spec.id] = spec
            self._by_name[spec.name] = spec

        self._verify_fallback_acyclic()

    def _verify_fallback_acyclic(self) -> None:
        """DFS cycle check across the entire fallback graph."""
        # 0 = unvisited, 1 = visiting, 2 = visited
        state: dict[str, int] = {eid: 0 for eid in self._by_id}

        def dfs(eid: str, chain: list[str]) -> None:
            state[eid] = 1
            fb = self._by_id[eid].fallback_id
            if fb is not None:
                if fb not in self._by_id:
                    raise SchemaValidationError(f"Effect {eid} points to unknown fallback_id {fb!r}")
                if state[fb] == 1:
                    raise SchemaValidationError(f"Fallback cycle detected: {' -> '.join(chain + [fb])}")
                if state[fb] == 0:
                    dfs(fb, chain + [fb])
            state[eid] = 2

        for eid in sorted(self._by_id):
            if state[eid] == 0:
                dfs(eid, [eid])

    def get(self, effect_id_or_name: str) -> EffectSpec:
        if effect_id_or_name in self._by_id:
            return self._by_id[effect_id_or_name]
        if effect_id_or_name in self._by_name:
            return self._by_name[effect_id_or_name]
        raise SchemaValidationError(f"Unknown effect: {effect_id_or_name!r}")

    def has(self, effect_id: str) -> bool:
        return effect_id in self._by_id

    def all_specs(self) -> list[EffectSpec]:
        return [self._by_id[k] for k in sorted(self._by_id.keys())]

    def count_by_family(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for spec in self.all_specs():
            counts[spec.family] = counts.get(spec.family, 0) + 1
        return counts

    def resolve_engine(
        self,
        spec: EffectSpec,
        engine_caps: dict[str, Any],
        preferred_engine: str = "auto",
    ) -> str | None:
        """Pick the best available licensed engine for this spec."""
        available = {
            name for name, info in engine_caps.items()
            if isinstance(info, dict) and info.get("available")
        }
        if preferred_engine != "auto":
            if preferred_engine in spec.supported_engines and preferred_engine in available:
                return preferred_engine
        # Honor primary engine if available, then supported fallback engines
        if spec.engine in available:
            return spec.engine
        for eng in spec.supported_engines:
            if eng in available:
                return eng
        return None


def load_effects(custom_specs: list[EffectSpec] | None = None) -> EffectRegistry:
    """Load and verify the ML-01 effect registry."""
    specs = custom_specs if custom_specs is not None else build_effect_catalog()
    return EffectRegistry(specs)
