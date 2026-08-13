"""Package the v8 release: one ZIP per task plus one combined ZIP.

    python3 _dev/package.py

Per-task ZIPs contain exactly the Harbor task tree. The combined ZIP adds the
governance state, the validation and audit reports, and the development
harness, so the whole evidence chain travels with the tasks.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
TASKS = ["alzheimer-abeta42-v8", "parkinson-alpha-synuclein-v8", "alzheimer-tau-v8",
         "als-ftd-tdp43-v8", "huntington-htt-polyq-v8"]
SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", "dist"}
SKIP_SUFFIX = {".log", ".pyc", ".pyo"}
SKIP_NAMES = {".DS_Store"}


def keep(p: Path) -> bool:
    return (p.is_file() and not SKIP_DIRS & set(p.parts)
            and p.suffix not in SKIP_SUFFIX and p.name not in SKIP_NAMES)


def add(zf: zipfile.ZipFile, src: Path, arc: str) -> int:
    n = 0
    for p in sorted(src.rglob("*")):
        if keep(p):
            zf.write(p, f"{arc}/{p.relative_to(src)}")
            n += 1
    return n


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    DIST.mkdir(exist_ok=True)
    for old in DIST.glob("*.zip"):
        old.unlink()
    made = []

    for t in TASKS:
        z = DIST / f"{t}.zip"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            n = add(zf, ROOT / t, t)
        made.append((z, n))

    combined = DIST / "neurofold-harbor-v8-all-tasks.zip"
    with zipfile.ZipFile(combined, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        n = 0
        for t in TASKS:
            n += add(zf, ROOT / t, f"neurofold-harbor-v8/{t}")
        n += add(zf, ROOT / "agentic", "neurofold-harbor-v8/agentic")
        n += add(zf, ROOT / "_dev", "neurofold-harbor-v8/_dev")
        for f in ("README.md",):
            if (ROOT / f).exists():
                zf.write(ROOT / f, f"neurofold-harbor-v8/{f}")
                n += 1
    made.append((combined, n))

    lines, manifest = [], {}
    for z, n in made:
        d = sha256(z)
        mb = z.stat().st_size / 1e6
        manifest[z.name] = {"sha256": d, "bytes": z.stat().st_size, "files": n}
        lines.append(f"{d}  {z.name}")
        print(f"{z.name:44s} {mb:6.2f} MB  {n:4d} files")
    (DIST / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    (DIST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {DIST}/SHA256SUMS.txt and manifest.json")


if __name__ == "__main__":
    main()
