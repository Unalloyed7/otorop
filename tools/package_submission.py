"""Собрать конкурсный ZIP с корневыми модулями, без веба, данных и ключей."""
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("schemas.py", "llm_client.py", "assistant.py", "search_adapter.py", "documents.py", "retrieval.py", "solution.py")


def package(output: Path, *, root: Path = ROOT, extra_modules: tuple[str, ...] = ()) -> Path:
    names = sorted(set((*MODULES, "requirements.txt", *extra_modules)))
    files = {}
    for name in names:
        if Path(name).name != name or (name != "requirements.txt" and not name.endswith(".py")):
            raise ValueError("Дополнительные модули должны быть корневыми .py")
        data = (root / name).read_bytes()
        if name.endswith(".py"):
            compile(data, name, "exec")
        files[name] = data
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "psb-submission.zip")
    parser.add_argument("--extra-module", action="append", default=[])
    args = parser.parse_args()
    path = package(args.output, extra_modules=tuple(args.extra_module))
    print(f"Создан {path}: {path.stat().st_size} байт")


if __name__ == "__main__":
    main()
