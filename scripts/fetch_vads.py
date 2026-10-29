"""Download the VAD bench's model files into models/vad/.

The VAD counterpart of fetch_models.py, and just as idempotent:

    .venv/bin/python scripts/fetch_vads.py [vad-id ...]
    .venv/bin/python scripts/fetch_vads.py --verify      # re-hash what's on disk

The file list lives in vad_candidates.json (each entry's "files"), not here, so
a VAD's weights and the code that loads them are declared in one place. Every
file comes from its publisher's own URL (a release tag or a pinned commit, never
a moving branch) and is checked against a pinned SHA-256 before it is renamed
into place. models/vad/ is gitignored; no weights live in this repo.

Two families are not plain URLs:

- ai-coustics models only come from the vendor CDN through the SDK's own
  Model.download(), which picks the newest build compatible with the installed
  aic-sdk. The pin is the file name (it carries a build suffix) plus its hash,
  so a new upstream build shows up here as a mismatch instead of silently
  changing what "aic-vad-ms-2.1" means.
- TEN, Earshot, WebRTC and lk-inference ship their weights inside their pip
  packages, so there is nothing to fetch.
"""

import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
REGISTRY = ROOT / "vad_candidates.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _aic_download(model_id: str, dst: Path) -> Path:
    """Model.download into a temp dir, then move into place only if it is the
    pinned build — so a republished model never lands under models/vad/."""
    import aic_sdk as aic

    with tempfile.TemporaryDirectory() as tmp:
        got = Path(aic.Model.download(model_id, tmp))
        if got.name != dst.name:
            raise RuntimeError(f"ai-coustics now serves {got.name} for {model_id}, "
                               f"pinned {dst.name}")
        tmp_dst = dst.with_suffix(dst.suffix + ".part")
        shutil.move(str(got), tmp_dst)
    return tmp_dst


def main(argv: list[str]) -> None:
    only = [a for a in argv if not a.startswith("--")]
    verify_only = "--verify" in argv
    entries = json.loads(REGISTRY.read_text())
    todo: dict[str, dict] = {}
    for e in entries:
        if only and e["id"] not in only:
            continue
        todo.update(e.get("files") or {})
        # files only a parity check reads (not part of the VAD's weights, so
        # fetching one never invalidates stored probabilities)
        todo.update(e.get("verify_files") or {})
    if not todo:
        sys.exit(f"nothing to fetch for {only or 'all'}")

    bad = []
    for rel, spec in todo.items():
        dst = MODELS / rel
        want = spec["sha256"]
        if verify_only or dst.exists():
            if not dst.exists():
                print(f"MISSING  {rel}")
                bad.append(rel)
            elif sha256(dst) != want:
                print(f"MISMATCH {rel}\n         want {want}\n         got  {sha256(dst)}")
                bad.append(rel)
            else:
                print(f"{'ok  ' if verify_only else 'have'}  {rel} ({dst.stat().st_size / 1e6:.1f} MB)")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            if "aic_model_id" in spec:
                print(f"get   {rel} <- ai-coustics CDN ({spec['aic_model_id']})")
                tmp = _aic_download(spec["aic_model_id"], dst)
            else:
                print(f"get   {rel} <- {spec['url']}")
                tmp = dst.with_suffix(dst.suffix + ".part")
                urllib.request.urlretrieve(spec["url"], tmp)  # noqa: S310 - pinned https URLs
        except Exception as e:  # noqa: BLE001 - one failed file must not stop the rest
            print(f"FAILED   {rel}: {e}")
            bad.append(rel)
            continue
        # verify BEFORE the rename, as fetch_models.py does: a wrong file must
        # never sit under its real name, or the next run "have"s it forever
        got = sha256(tmp)
        if got != want:
            print(f"MISMATCH {rel}\n         want {want}\n         got  {got}")
            tmp.unlink()
            bad.append(rel)
            continue
        tmp.rename(dst)
        print(f"      done ({dst.stat().st_size / 1e6:.1f} MB)")

    if bad:
        sys.exit(f"\n{len(bad)} file(s) failed: {', '.join(bad)}\n"
                 "A hash mismatch means upstream republished the file: compare against the "
                 "probabilities already stored under data/runs/ before trusting new ones.")


if __name__ == "__main__":
    main(sys.argv[1:])
