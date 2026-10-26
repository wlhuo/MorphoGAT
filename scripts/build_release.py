"""Build a clean source archive with a SHA-256 checksum and file inventory."""
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ["README.md", "README_zh.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "CITATION.cff",
         "pyproject.toml", "requirements.txt", "requirements-dev.txt", "requirements-analysis.txt",
         ".gitignore", "environment.yml", "MANIFEST.in"]
DIRECTORIES = ["morphogat", "configs", "examples", "scripts", "tests", "docs", "notebooks", ".github"]


def main():
    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    source_files = [ROOT / name for name in FILES]
    for name in DIRECTORIES:
        source_files.extend(path for path in (ROOT / name).rglob("*")
                            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
    source_files = sorted(set(source_files))
    missing = [str(path.relative_to(ROOT)) for path in source_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    archive = output / "MorphoGAT-code.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in source_files:
            bundle.write(path, "MorphoGAT/" + path.relative_to(ROOT).as_posix())
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output / "MorphoGAT-code.zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    inventory = "\n".join(path.relative_to(ROOT).as_posix() for path in source_files) + "\n"
    (output / "MorphoGAT-files.txt").write_text(inventory, encoding="utf-8")
    print(f"{archive}: {len(source_files)} files, {archive.stat().st_size} bytes")
    print(digest)


if __name__ == "__main__":
    main()
