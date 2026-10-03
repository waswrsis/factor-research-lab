#!/usr/bin/env python3
"""Stopped-project ZIP transfer with a per-file manifest and external checksum.

Stop all writers first: this tool detects changes, but is not a database snapshot.
Uses only Python's standard library. Existing destination directories are refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import unicodedata
import zipfile
from pathlib import Path

MANIFEST = "_TRANSFER_MANIFEST.json"
CHUNK = 1024 * 1024
EXCLUDED_DIRS = {
    ".venv", ".bootstrap", ".tools", ".cache", ".git", "node_modules", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".nox",
}
SECRET_DIRS = {".ssh", ".aws", ".gnupg", "privatekeys", "private_keys", ".codex"}
SECRET_NAMES = {
    "auth.json", "credentials.json", "credential.json", "token.json", "tokens.json",
    "secrets.json", "secrets.toml", "secrets.yaml", "secrets.yml", ".netrc",
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "privatekey", "private_key",
}
RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10))}


def _linked(path: Path) -> bool:
    value = path.lstat()
    return stat.S_ISLNK(value.st_mode) or bool(
        getattr(value, "st_file_attributes", 0) & 0x400)  # Windows reparse points


def _safe_name(name: str) -> None:
    if not isinstance(name, str) or not name or "\\" in name or name.startswith("/"):
        raise ValueError("Unsafe archive path")
    parts = name.split("/")
    for part in parts:
        if (part in {"", ".", ".."} or part.endswith((".", " "))
                or any(ord(c) < 32 or c in '<>:"|?*' for c in part)
                or part.split(".")[0].upper() in RESERVED):
            raise ValueError("Unsafe archive path")


def _secret(name: str) -> bool:
    lower = name.casefold()
    return (lower in SECRET_NAMES or lower in SECRET_DIRS or lower == ".env"
            or lower.startswith((".env.", "privatekey.", "private_key.", "credentials.", "secrets."))
            or lower.endswith((".pem", ".key", ".p12", ".pfx")))


def _signature(path: Path) -> tuple:
    value = path.lstat()
    if _linked(path) or not stat.S_ISREG(value.st_mode):
        raise ValueError(f"Non-regular file or external reference refused: {path}")
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns, value.st_mode)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _scan(root: Path, ignored_paths: set[Path], excludes: tuple[str, ...]) -> dict:
    files = {}
    folded = set()

    def visit(directory: Path) -> None:
        for entry in sorted(directory.iterdir(), key=lambda p: p.name):
            relative = entry.relative_to(root).as_posix()
            lower = entry.name.casefold()
            if (relative == MANIFEST or entry in ignored_paths or lower in EXCLUDED_DIRS or _secret(entry.name)
                    or relative == "work/migration_export" or relative.startswith("work/migration_export/")
                    or any(relative == item or relative.startswith(item + "/") for item in excludes)):
                continue
            if _linked(entry):
                raise ValueError(f"Symlink/junction refused: {relative}. Use --exclude for this reference.")
            _safe_name(relative)
            key = unicodedata.normalize("NFC", relative).casefold()
            if key in folded or key == MANIFEST.casefold():
                raise ValueError(f"Duplicate or reserved transfer path: {relative}")
            folded.add(key)
            if entry.is_dir():
                visit(entry)
            else:
                files[relative] = _signature(entry)

    visit(root)
    return files


def pack(root: Path, archive: Path, excludes: tuple[str, ...] = ()) -> dict:
    root = root.absolute()
    if not root.is_dir() or _linked(root):
        raise ValueError("ROOT must be a real project directory, not a symlink/junction")
    root = root.resolve()
    archive = archive.absolute()
    if archive.parent.exists() and _linked(archive.parent):
        raise ValueError("Archive parent cannot be a symlink/junction")
    archive = archive.resolve()
    checksum = Path(str(archive) + ".sha256")
    partial = Path(str(archive) + ".part")
    if any(os.path.lexists(p) for p in (archive, checksum, partial)):
        raise ValueError("Archive, checksum and .part paths must all be new")
    for item in excludes:
        _safe_name(item)
    archive.parent.mkdir(parents=True, exist_ok=True)
    ignored = {archive, checksum, partial}
    before = _scan(root, ignored, excludes)
    records = {}
    for relative, signature in before.items():
        path = root / relative
        digest = _hash(path)
        if _signature(path) != signature:
            raise ValueError(f"Source changed during preflight: {relative}")
        records[relative] = {"sha256": digest, "size": signature[2],
                             "mode": stat.S_IMODE(signature[5]) & 0o777}
    if _scan(root, ignored, excludes) != before:
        raise ValueError("Source inventory changed during preflight; stop writers and retry")
    with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for relative, signature in before.items():
            path = root / relative
            if _signature(path) != signature:
                raise ValueError(f"Source changed before packing: {relative}")
            info = zipfile.ZipInfo(relative)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | records[relative]["mode"]) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            digest = hashlib.sha256()
            with path.open("rb") as source, bundle.open(info, "w", force_zip64=True) as output:
                for block in iter(lambda: source.read(CHUNK), b""):
                    digest.update(block)
                    output.write(block)
            if digest.hexdigest() != records[relative]["sha256"] or _signature(path) != signature:
                raise ValueError(f"Source changed while packing: {relative}")
        manifest = {"format": "factorresearch-transfer-v1", "files": records,
                    "note": "Stopped-project snapshot; known credentials, caches and root transfer receipt excluded."}
        bundle.writestr(MANIFEST, json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    if _scan(root, ignored, excludes) != before:
        raise ValueError("Source inventory changed after packing; .part retained for diagnosis")
    if archive.exists():
        raise ValueError("Archive destination appeared during packing; .part retained")
    partial.replace(archive)
    digest = _hash(archive)
    with checksum.open("x", encoding="utf-8") as handle:
        handle.write(f"{digest}  {archive.name}\n")
    return {"archive": str(archive), "checksum": str(checksum), "files": len(records),
            "bytes": sum(row["size"] for row in records.values()), "sha256": digest}


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate manifest key")
        result[key] = value
    return result


def unpack(archive: Path, destination: Path) -> dict:
    destination = destination.absolute()
    if os.path.lexists(destination):
        raise ValueError("DEST must not exist; use a new restoration directory")
    archive = archive.absolute()
    if _linked(archive):
        raise ValueError("Archive cannot be a symlink/junction")
    checksum = Path(str(archive) + ".sha256")
    if _linked(checksum) or checksum.stat().st_size > 4096:
        raise ValueError("Invalid checksum sidecar")
    content = checksum.read_text("utf-8").strip().split()
    if not content or not re.fullmatch(r"[0-9a-f]{64}", content[0]):
        raise ValueError("Invalid archive SHA-256 sidecar")
    expected = content[0]
    signature = _signature(archive)
    if _hash(archive) != expected:
        raise ValueError("Archive SHA-256 mismatch")
    with zipfile.ZipFile(archive, "r") as bundle:
        members = {}
        folded = set()
        for info in bundle.infolist():
            # ZipInfo may normalize backslashes or truncate NULs on Windows.
            # Validate its untouched header name before trusting filename.
            _safe_name(info.orig_filename)
            _safe_name(info.filename)
            key = unicodedata.normalize("NFC", info.filename).casefold()
            mode = info.external_attr >> 16
            if key in folded:
                raise ValueError("Duplicate archive member")
            if info.is_dir() or stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                raise ValueError("Archive symlink or non-regular member refused")
            folded.add(key)
            members[info.filename] = info
        if MANIFEST not in members or members[MANIFEST].file_size > 64 * CHUNK:
            raise ValueError("Transfer manifest missing or too large")
        manifest = json.loads(bundle.read(MANIFEST), object_pairs_hook=_no_duplicate_keys)
        if not isinstance(manifest, dict) or manifest.get("format") != "factorresearch-transfer-v1":
            raise ValueError("Unsupported transfer manifest")
        records = manifest.get("files")
        if not isinstance(records, dict) or set(records) != set(members) - {MANIFEST}:
            raise ValueError("Manifest and archive file inventories differ")
        record_keys = {unicodedata.normalize("NFC", name).casefold() for name in records}
        for relative, row in records.items():
            _safe_name(relative)
            if (not isinstance(row, dict) or set(row) != {"sha256", "size", "mode"}
                    or not isinstance(row["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
                    or type(row["size"]) is not int or row["size"] < 0
                    or row["size"] != members[relative].file_size
                    or type(row["mode"]) is not int or not 0 <= row["mode"] <= 0o777):
                raise ValueError("Invalid manifest file record")
            parts = relative.split("/")
            if any(unicodedata.normalize("NFC", "/".join(parts[:i])).casefold() in record_keys
                   for i in range(1, len(parts))):
                raise ValueError("Conflicting file and directory archive paths")
        destination.mkdir(parents=True)
        destination = destination.resolve()
        for relative, row in records.items():
            target = destination / relative
            if not target.resolve().is_relative_to(destination):
                raise ValueError("Extraction path escapes destination")
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            count = 0
            with bundle.open(members[relative]) as source, target.open("xb") as output:
                for block in iter(lambda: source.read(CHUNK), b""):
                    count += len(block)
                    if count > row["size"]:
                        raise ValueError("Extracted member exceeds manifest size")
                    digest.update(block)
                    output.write(block)
            if count != row["size"] or digest.hexdigest() != row["sha256"]:
                raise ValueError("Extracted file SHA-256 mismatch; destination retained for diagnosis")
            if os.name != "nt":
                target.chmod(row["mode"])
        (destination / MANIFEST).write_bytes(bundle.read(MANIFEST))
    if _signature(archive) != signature or _hash(archive) != expected:
        raise ValueError("Archive changed during extraction; destination retained for diagnosis")
    return {"destination": str(destination), "files": len(records), "sha256": expected}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    packing = sub.add_parser("pack", help="Stop writers, then create a new ZIP plus ZIP.sha256")
    packing.add_argument("root", type=Path, metavar="ROOT")
    packing.add_argument("archive", type=Path, metavar="ARCHIVE")
    packing.add_argument("--exclude", action="append", default=[], metavar="RELATIVE_PATH",
                         help="Exclude this project-relative file/directory; repeat as needed")
    extraction = sub.add_parser("unpack", help="Verify ZIP.sha256 and restore into a nonexistent directory")
    extraction.add_argument("archive", type=Path, metavar="ARCHIVE")
    extraction.add_argument("destination", type=Path, metavar="DEST")
    args = parser.parse_args(argv)
    try:
        result = (pack(args.root, args.archive, tuple(args.exclude)) if args.command == "pack"
                  else unpack(args.archive, args.destination))
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        print(f"Transfer refused: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
