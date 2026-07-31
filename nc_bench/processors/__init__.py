"""Processor registry: build chains from candidates.json specs."""

from __future__ import annotations

from .aic import AICProcessor, aic_available
from .base import Passthrough, Processor
from .dtln import DTLNProcessor, dtln_available
from .fastenhancer import FastEnhancerProcessor, fastenhancer_available
from .hecttor import HecttorProcessor, hecttor_available
from .hush import HushProcessor, hush_available
from .rnnoise import RNNoiseProcessor, rnnoise_available
from .spec_onnx import SpecOnnxProcessor, spec_onnx_available

# proc name -> (availability checker, factory); both take the candidate's spec
_REGISTRY = {
    "hecttor": (
        lambda spec: hecttor_available(),
        lambda spec: HecttorProcessor(
            model=spec.get("model"),
            enhancer_weight=spec.get("enhancer_weight"),
            sample_rate=spec.get("sample_rate"),
            chunk_ms=spec.get("chunk_ms"),
        ),
    ),
    "dtln": (lambda spec: dtln_available(), lambda spec: DTLNProcessor()),
    # DPDFNet / GTCRN / UL-UNAS all share the frame-spectral streaming interface
    "specnc": (
        lambda spec: spec_onnx_available(spec.get("model")),
        lambda spec: SpecOnnxProcessor(spec["model"]),
    ),
    "fastenhancer": (
        lambda spec: fastenhancer_available(spec.get("model")),
        lambda spec: FastEnhancerProcessor(spec["model"]),
    ),
    "rnnoise": (
        lambda spec: rnnoise_available(spec.get("model")),
        lambda spec: RNNoiseProcessor(spec.get("model")),
    ),
    # the only open background-speaker suppressor (DFN3 retrained, 16 kHz native)
    "hush": (
        hush_available,
        lambda spec: HushProcessor(atten_lim_db=spec.get("atten_lim_db")),
    ),
    "aic": (
        lambda spec: aic_available(),
        lambda spec: AICProcessor(
            model_id=spec.get("model"),
            enhancement_level=spec.get("enhancement_level"),
        ),
    ),
}


def proc_available(spec: dict) -> tuple[bool, str]:
    name = spec.get("proc", "")
    if name not in _REGISTRY:
        return False, f"unknown processor '{name}'"
    return _REGISTRY[name][0](spec)


def chain_available(chain: list[dict]) -> tuple[bool, str]:
    for spec in chain:
        ok, why = proc_available(spec)
        if not ok:
            return False, why
    return True, ""


def build_chain(chain: list[dict]) -> list[Processor]:
    """Instantiate the processors for one candidate. Empty chain = passthrough."""
    if not chain:
        return [Passthrough()]
    return [_REGISTRY[spec["proc"]][1](spec) for spec in chain]
