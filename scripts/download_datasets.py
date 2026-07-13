"""Download and safely extract the frozen BVTSLD and TT100K datasets."""
from __future__ import annotations

import argparse
import shutil
import stat
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3
USER_AGENT = "wvc2026-deteccao-placas-reproduction/1.0"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    archive_name: str
    expected_bytes: int
    extract_root: Path
    payload: Path
    markers: tuple[Path, ...]
    minimum_free_gib: int
    license_name: str
    license_url: str
    source_url: str


SPECS = {
    "bvtsld": DatasetSpec(
        name="BVTSLD v2",
        url="https://data.mendeley.com/public-api/zip/jbpsr4fvg9/download/2",
        archive_name="bvtsld-v2.zip",
        expected_bytes=4_142_062_627,
        extract_root=ROOT / "datasets" / "bvtsld",
        payload=Path("Brazilian Vertical Traffic Signs and Lights Dataset"),
        markers=(Path("images"), Path("annotations")),
        minimum_free_gib=12,
        license_name="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        source_url="https://data.mendeley.com/datasets/jbpsr4fvg9/2",
    ),
    "tt100k": DatasetSpec(
        name="Tsinghua-Tencent 100K (2016 annotations)",
        url="https://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/data.zip",
        archive_name="tt100k-2016-data.zip",
        expected_bytes=19_152_969_603,
        extract_root=ROOT / "datasets" / "tt100k",
        payload=Path("data"),
        markers=(Path("annotations.json"), Path("train"), Path("test"), Path("other")),
        minimum_free_gib=100,
        license_name="CC BY-NC (non-commercial; version not stated by the source)",
        license_url="https://cg.cs.tsinghua.edu.cn/traffic-sign/",
        source_url="https://cg.cs.tsinghua.edu.cn/traffic-sign/",
    ),
}


def installed(spec: DatasetSpec) -> bool:
    base = spec.extract_root / spec.payload
    return all((base / marker).exists() for marker in spec.markers)


def request(url: str, start: int | None = None) -> urllib.request.Request:
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if start is not None:
        headers["Range"] = f"bytes={start}-"
    return urllib.request.Request(url, headers=headers)


def remote_size(spec: DatasetSpec) -> int:
    with urllib.request.urlopen(request(spec.url, 0), timeout=60) as response:
        content_range = response.headers.get("Content-Range", "")
        if "/" in content_range:
            return int(content_range.rsplit("/", 1)[1])
        return int(response.headers["Content-Length"])


def check_source(spec: DatasetSpec) -> None:
    size = remote_size(spec)
    if size != spec.expected_bytes:
        raise RuntimeError(
            f"upstream size changed for {spec.name}: {size} != {spec.expected_bytes}"
        )
    print(f"{spec.name}: source reachable, {size / GIB:.2f} GiB")


def ensure_disk_space(spec: DatasetSpec) -> None:
    spec.extract_root.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(spec.extract_root.parent).free
    required = spec.minimum_free_gib * GIB
    if free < required:
        raise RuntimeError(
            f"{spec.name} requires at least {spec.minimum_free_gib} GiB free; "
            f"available: {free / GIB:.1f} GiB"
        )


def download(spec: DatasetSpec, archive_dir: Path) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    final = archive_dir / spec.archive_name
    partial = final.with_suffix(final.suffix + ".part")
    if final.exists():
        if final.stat().st_size == spec.expected_bytes:
            print(f"reusing archive: {final}")
            return final
        raise RuntimeError(f"invalid completed archive size: {final}")

    start = partial.stat().st_size if partial.exists() else 0
    mode = "ab" if start else "wb"
    print(f"downloading {spec.name}: {spec.url}")
    if start:
        print(f"resuming at {start / GIB:.2f} GiB")
    with urllib.request.urlopen(request(spec.url, start or None), timeout=120) as response:
        if start and response.status != 206:
            print("server did not honor Range; restarting download")
            start, mode = 0, "wb"
        downloaded = start
        last_report = time.monotonic()
        with partial.open(mode) as handle:
            while chunk := response.read(8 * 1024 * 1024):
                handle.write(chunk)
                downloaded += len(chunk)
                if time.monotonic() - last_report >= 5:
                    print(
                        f"  {downloaded / GIB:.2f}/{spec.expected_bytes / GIB:.2f} GiB "
                        f"({100 * downloaded / spec.expected_bytes:.1f}%)",
                        flush=True,
                    )
                    last_report = time.monotonic()
    if partial.stat().st_size != spec.expected_bytes:
        raise RuntimeError(
            f"incomplete download: {partial.stat().st_size} != {spec.expected_bytes}"
        )
    partial.replace(final)
    return final


def validate_zip_members(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if root != target and root not in target.parents:
            raise RuntimeError(f"unsafe path in archive: {member.filename}")
        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"symbolic link rejected in archive: {member.filename}")


def extract(spec: DatasetSpec, archive_path: Path, force: bool) -> None:
    payload = spec.extract_root / spec.payload
    if payload.exists() and force:
        shutil.rmtree(payload)
    spec.extract_root.mkdir(parents=True, exist_ok=True)
    print(f"extracting {archive_path} -> {spec.extract_root}")
    with zipfile.ZipFile(archive_path) as archive:
        validate_zip_members(archive, spec.extract_root)
        archive.extractall(spec.extract_root)
    if not installed(spec):
        raise RuntimeError(
            f"unexpected archive layout; expected dataset at {payload}. "
            f"Inspect {spec.extract_root} before continuing."
        )


def install(
    spec: DatasetSpec,
    archive_dir: Path,
    accept_license: bool,
    force: bool,
    keep_archive: bool,
    dry_run: bool,
) -> None:
    print(f"\n== {spec.name} ==")
    print(f"source: {spec.source_url}")
    print(f"license: {spec.license_name} ({spec.license_url})")
    print(f"destination: {spec.extract_root / spec.payload}")
    if installed(spec) and not force:
        print("already installed")
        return
    if dry_run:
        print(f"would download {spec.expected_bytes / GIB:.2f} GiB and extract safely")
        return
    if not accept_license:
        raise RuntimeError("review the license and rerun with --accept-license")
    ensure_disk_space(spec)
    check_source(spec)
    archive_path = download(spec, archive_dir)
    extract(spec, archive_path, force)
    if not keep_archive:
        archive_path.unlink()
    print("installation complete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("bvtsld", "tt100k", "all"), required=True)
    parser.add_argument("--accept-license", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-extract an installed dataset")
    parser.add_argument("--keep-archive", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true", help="Check official URLs and frozen sizes only")
    parser.add_argument("--archive-dir", type=Path, default=ROOT / "datasets" / ".archives")
    args = parser.parse_args()

    names = tuple(SPECS) if args.dataset == "all" else (args.dataset,)
    try:
        for name in names:
            spec = SPECS[name]
            if args.check:
                check_source(spec)
            else:
                install(
                    spec, args.archive_dir, args.accept_license, args.force,
                    args.keep_archive, args.dry_run,
                )
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
