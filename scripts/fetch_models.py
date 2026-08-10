"""Download the model files the candidates need into models/.

Idempotent (skips what's already there) and safe to re-run:

    .venv/bin/python scripts/fetch_models.py [name ...]
    .venv/bin/python scripts/fetch_models.py --verify     # re-hash what's on disk

Nothing is trained or converted locally and nothing is redistributed in this
repo — every entry below is a published release artifact fetched from the
publisher's own URL, and models/ is gitignored. See README "Models and licences"
for who owns what.

Each entry carries a SHA-256, checked after download and on --verify. That is
not paranoia: microsoft/DNS-Challenge ships two different files both named
sig_bak_ovr.onnx (DNSMOS/ and pDNSMOS/, three bytes apart), so a plausible-looking
path silently swaps the scoring model and every DNSMOS number in data/runs/ stops
meaning what it did. A hash makes "same model as the stored results" checkable
rather than assumed.
"""

import hashlib
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
_DTLN = "https://raw.githubusercontent.com/breizhn/DTLN/master/pretrained_model"
_DNSMOS = "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS"
_HUSH_BUNDLE = "advanced_dfnet16k_model_best_onnx.tar.gz"
_HUSH_LIB = {"Darwin": "libweya_nc.dylib", "Windows": "weya_nc.dll"}.get(
    platform.system(), "libweya_nc.so"
)

# local path under models/ -> (url, sha256)
FILES: dict[str, tuple[str, str]] = {
    # DPDFNet (Ceva, Apache-2.0) — 8 kHz pair is the one that matters for PSTN
    "dpdfnet/dpdfnet2_8khz.onnx": (
        f"{_HF_DPDFNET}/dpdfnet2_8khz.onnx",
        "6218f1dbd6e4bac5768c63b7d899fe7b84b3788f2a35c4e246d4ab0946165c5d",
    ),
    "dpdfnet/dpdfnet8_8khz.onnx": (
        f"{_HF_DPDFNET}/dpdfnet8_8khz.onnx",
        "c061bcc56b803fa2fa97d448a45db6d966f7d17aff1304e464455d748745ea62",
    ),
    "dpdfnet/baseline.onnx": (
        f"{_HF_DPDFNET}/baseline.onnx",
        "371d26182aff0e1e0d31354e24c81f79cd57458f5a7dd003fc10a7ebf64255e0",
    ),
    "dpdfnet/dpdfnet2.onnx": (
        f"{_HF_DPDFNET}/dpdfnet2.onnx",
        "4f0ee28935b4a32abecc717d745416976565834d839601acf43031094b4dc94c",
    ),
    "dpdfnet/dpdfnet8.onnx": (
        f"{_HF_DPDFNET}/dpdfnet8.onnx",
        "899d4f23f3ff86edbffa8c537e4bcbdc49da1b4e84e0ef390611e0604a3b26cb",
    ),
    # GTCRN (MIT) — sherpa-onnx's export of Xiaobin-Rong/gtcrn (carries STFT metadata)
    "gtcrn/gtcrn_simple.onnx": (
        f"{_SHERPA}/gtcrn_simple.onnx",
        "e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534",
    ),
    # UL-UNAS (MIT) — streaming export straight from the paper's repo
    "ulunas/ulunas_stream_simple.onnx": (
        "https://raw.githubusercontent.com/Xiaobin-Rong/ul-unas/main/"
        "ulunas_onnx/onnx_models/ulunas_stream_simple.onnx",
        "f2e804d54d6a88f4f82f44d86c9f1cf646db2509bfca935cfbfc5fcd8cbfac3b",
    ),
    # FastEnhancer (MIT) — DNS-trained 16 kHz wav2wav (STFT inside the graph)
    "fastenhancer/fastenhancer_t_dns.onnx": (
        f"{_FE}/fastenhancer_t.onnx",
        "a64663672b31bd1445502ee23328d291af13fb312e11bb574c010e7261a5ee7d",
    ),
    "fastenhancer/fastenhancer_s_dns.onnx": (
        f"{_FE}/fastenhancer_s.onnx",
        "d7c937d6b475dbe89a6e6a69853b685fe62d629f4c96077559c556ac6f8aaa2d",
    ),
    "fastenhancer/fastenhancer_l_dns.onnx": (
        f"{_FE}/fastenhancer_l.onnx",
        "a211423e348707fba4952f2b868084a8d8880c6e8bd17333f4b1c0e87bd35b12",
    ),
    # RNNoise weights for ffmpeg's arnndn filter (no copyright claimed upstream)
    "rnnoise/sh.rnnn": (
        f"{_RNNOISE}/somnolent-hogwash-2018-09-01/sh.rnnn",
        "70bb6685eb0c2a1d18e2918dca3fbfbd39317010b1802eb1b6ea73a92f3fdec0",
    ),
    "rnnoise/bd.rnnn": (
        f"{_RNNOISE}/beguiling-drafter-2018-08-30/bd.rnnn",
        "ae3f7411e1e6a884f839a4a145c394408398f09854dbc1216ee02faafc98a17b",
    ),
    # Hush / Weya NC (Apache-2.0): model bundle + the prebuilt native lib that
    # does the DFN feature/gain DSP the ONNX graphs do not
    f"hush/{_HUSH_BUNDLE}": (
        f"{_HUSH}/models/{_HUSH_BUNDLE}",
        "45632ccaa82b71bb743d6caa7c78e983fe2f2790a3af7f6ec48e6ed7ba085df6",
    ),
    f"hush/{_HUSH_LIB}": (
        f"{_HUSH}/lib/{_HUSH_LIB}",
        # only the macOS arm64 build is pinned; the Linux/Windows builds are
        # fetched unverified because they have not been run here
        "4bcd38634000d456ad68db4b6ff97fe4a462542aa65ac5f910fbf374ca33fbaa"
        if platform.system() == "Darwin" and platform.machine() == "arm64"
        else "",
    ),
    # DTLN (MIT) — the two-stage pretrained pair
    "dtln/model_1.onnx": (
        f"{_DTLN}/model_1.onnx",
        "22b91cae3855e5a0620e66a917ca6c82c58db0e842c770f58d86751c5e8d4ae3",
    ),
    "dtln/model_2.onnx": (
        f"{_DTLN}/model_2.onnx",
        "e20c92f9233fccf29cddf86970d0d0161a03aebccc26d6f4d5639c4d5ec2e639",
    ),
    # DNSMOS P.835 (Microsoft DNS-Challenge, CC BY 4.0) — the SCORING model, not
    # a candidate. Must be DNSMOS/, never pDNSMOS/ (see the module docstring).
    "dnsmos/sig_bak_ovr.onnx": (
        f"{_DNSMOS}/sig_bak_ovr.onnx",
        "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(rel: str, dst: Path, want: str) -> bool:
    """True if the file is what we expect (or unpinned). Prints its own verdict."""
    if not want:
        print(f"      {rel}: no pinned hash for this platform, not verified")
        return True
    got = sha256(dst)
    if got == want:
        return True
    print(f"MISMATCH {rel}\n         want {want}\n         got  {got}")
    return False


def main(argv: list[str]) -> None:
    only = [a for a in argv if not a.startswith("--")]
    verify_only = "--verify" in argv

    todo = {k: v for k, v in FILES.items() if not only or any(o in k for o in only)}
    if not todo:
        sys.exit(f"nothing matched {only}; known: {sorted(FILES)}")

    bad = []
    for rel, (url, want) in todo.items():
        dst = MODELS / rel
        if verify_only:
            if not dst.exists():
                print(f"MISSING  {rel}")
                bad.append(rel)
            elif not verify(rel, dst, want):
                bad.append(rel)
            else:
                print(f"ok    {rel}")
            continue
        if dst.exists():
            print(f"have  {rel} ({dst.stat().st_size / 1e6:.1f} MB)")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"get   {rel} <- {url}")
        tmp = dst.with_suffix(dst.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https URLs above
        # verify BEFORE the rename: a corrupt or swapped file must never land
        # under its real name, or the next run "have"s it and never looks again
        if not verify(rel, tmp, want):
            tmp.unlink()
            bad.append(rel)
            continue
        tmp.rename(dst)
        print(f"      done ({dst.stat().st_size / 1e6:.1f} MB)")

    if bad:
        sys.exit(
            f"\n{len(bad)} file(s) failed: {', '.join(bad)}\n"
            "A hash mismatch means upstream republished the file. Compare against the "
            "run metadata in data/runs/ before trusting any score computed with it."
        )


if __name__ == "__main__":
    main(sys.argv[1:])
