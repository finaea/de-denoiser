"""Download the open NC model files the candidates need into models/.

Idempotent (skips what's already there) and safe to re-run. Everything here is
a published release artifact — nothing is trained or converted locally:

    .venv/bin/python scripts/fetch_models.py [name ...]

The big ONNX files are gitignored; this script is how a fresh clone gets them.
DTLN + DNSMOS are committed already, so they're not listed here.
"""

import platform
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

_HF_DPDFNET = "https://huggingface.co/Ceva-IP/DPDFNet/resolve/main/onnx"
_FE = "https://github.com/aask1357/fastenhancer/releases/download/onnx-dns-v1.0.0"
_SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models"
_RNNOISE = "https://raw.githubusercontent.com/GregorR/rnnoise-models/master"
_HUSH = "https://raw.githubusercontent.com/pulp-vision/Hush/main/deployment"
_HUSH_BUNDLE = "advanced_dfnet16k_model_best_onnx.tar.gz"
_HUSH_LIB = {"Darwin": "libweya_nc.dylib", "Windows": "weya_nc.dll"}.get(
    platform.system(), "libweya_nc.so"
)

# local path under models/ -> url
FILES: dict[str, str] = {
    # DPDFNet (Ceva, Apache-2.0) — 8 kHz pair is the one that matters for PSTN
    "dpdfnet/dpdfnet2_8khz.onnx": f"{_HF_DPDFNET}/dpdfnet2_8khz.onnx",
    "dpdfnet/dpdfnet8_8khz.onnx": f"{_HF_DPDFNET}/dpdfnet8_8khz.onnx",
    "dpdfnet/baseline.onnx": f"{_HF_DPDFNET}/baseline.onnx",
    "dpdfnet/dpdfnet2.onnx": f"{_HF_DPDFNET}/dpdfnet2.onnx",
    "dpdfnet/dpdfnet8.onnx": f"{_HF_DPDFNET}/dpdfnet8.onnx",
    # GTCRN — sherpa-onnx's export of Xiaobin-Rong/gtcrn (carries STFT metadata)
    "gtcrn/gtcrn_simple.onnx": f"{_SHERPA}/gtcrn_simple.onnx",
    # UL-UNAS — streaming export straight from the paper's repo
    "ulunas/ulunas_stream_simple.onnx": (
        "https://raw.githubusercontent.com/Xiaobin-Rong/ul-unas/main/"
        "ulunas_onnx/onnx_models/ulunas_stream_simple.onnx"
    ),
    # FastEnhancer — DNS-trained 16 kHz wav2wav (STFT inside the graph)
    "fastenhancer/fastenhancer_t_dns.onnx": f"{_FE}/fastenhancer_t.onnx",
    "fastenhancer/fastenhancer_s_dns.onnx": f"{_FE}/fastenhancer_s.onnx",
    "fastenhancer/fastenhancer_l_dns.onnx": f"{_FE}/fastenhancer_l.onnx",
    # RNNoise weights for ffmpeg's arnndn filter
    "rnnoise/sh.rnnn": f"{_RNNOISE}/somnolent-hogwash-2018-09-01/sh.rnnn",
    "rnnoise/bd.rnnn": f"{_RNNOISE}/beguiling-drafter-2018-08-30/bd.rnnn",
    # Hush: model bundle + the prebuilt native lib that does the DFN DSP
    f"hush/{_HUSH_BUNDLE}": f"{_HUSH}/models/{_HUSH_BUNDLE}",
    f"hush/{_HUSH_LIB}": f"{_HUSH}/lib/{_HUSH_LIB}",
}


def main(only: list[str]) -> None:
    todo = {k: v for k, v in FILES.items() if not only or any(o in k for o in only)}
    if not todo:
        sys.exit(f"nothing matched {only}; known: {sorted(FILES)}")
    for rel, url in todo.items():
        dst = MODELS / rel
        if dst.exists():
            print(f"have  {rel} ({dst.stat().st_size / 1e6:.1f} MB)")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"get   {rel} <- {url}")
        tmp = dst.with_suffix(dst.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https URLs above
        tmp.rename(dst)
        print(f"      done ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1:])
