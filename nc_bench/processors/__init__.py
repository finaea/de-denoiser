"""Processor registry: build chains from candidates.json specs."""

from __future__ import annotations

from .aic import AICProcessor, aic_available
from .base import Passthrough, Processor
from .dtln import DTLNProcessor, dtln_available
from .hecttor import HecttorProcessor, hecttor_available

# proc name -> (availability checker, factory)
_REGISTRY = {
    "hecttor": (
        hecttor_available,
        lambda spec: HecttorProcessor(
            model=spec.get("model"),
            enhancer_weight=spec.get("enhancer_weight"),
            sample_rate=spec.get("sample_rate"),
            chunk_ms=spec.get("chunk_ms"),
        ),
    ),
    "dtln": (dtln_available, lambda spec: DTLNProcessor()),
    "aic": (
        aic_available,
        lambda spec: AICProcessor(
            model_id=spec.get("model"),
            enhancement_level=spec.get("enhancement_level"),
        ),
    ),
}


def proc_available(name: str) -> tuple[bool, str]:
    if name not in _REGISTRY:
        return False, f"unknown processor '{name}'"
    return _REGISTRY[name][0]()


def chain_available(chain: list[dict]) -> tuple[bool, str]:
    for spec in chain:
        ok, why = proc_available(spec["proc"])
        if not ok:
            return False, why
    return True, ""


def build_chain(chain: list[dict]) -> list[Processor]:
    """Instantiate the processors for one candidate. Empty chain = passthrough."""
    if not chain:
        return [Passthrough()]
    return [_REGISTRY[spec["proc"]][1](spec) for spec in chain]
