from pathlib import Path
import zipfile

import pytest

from tools.package_submission import MODULES, package


def test_submission_contains_only_root_runtime_files(tmp_path):
    output = package(tmp_path / "submission.zip")
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {*MODULES, "requirements.txt"}
        assert all("/" not in name for name in archive.namelist())
        assert archive.read("requirements.txt") == Path("requirements.txt").read_bytes()
        for name in MODULES:
            compile(archive.read(name), name, "exec")
    assert output.read_bytes() == package(tmp_path / "again.zip").read_bytes()


def test_package_rejects_nested_module(tmp_path):
    with pytest.raises(ValueError):
        package(tmp_path / "submission.zip", extra_modules=("../secret.py",))
