import builtins
import json
from pathlib import Path

import pytest

from src.utils.paths import ensure_subpath, safe_write_json, sanitize_dirname


class TestPathsUtils:
    def test_sanitize_dirname_replaces_unsafe(self):
        name = "../weird name/with\\seps:*?<>|"
        s = sanitize_dirname(name)
        assert s  # non-empty
        assert "/" not in s and "\\" not in s
        for ch in [":", "*", "?", "<", ">", "|"]:
            assert ch not in s

    def test_ensure_subpath_within_root(self, tmp_path: Path):
        root = tmp_path / "root"
        root.mkdir()
        safe = ensure_subpath(root, Path("child"))
        assert str(safe).startswith(str(root.resolve()))

    def test_ensure_subpath_rejects_escape(self, tmp_path: Path):
        root = tmp_path / "root"
        root.mkdir()
        with pytest.raises(ValueError):
            ensure_subpath(root, Path("..") / "outside")

    def test_safe_write_json_success(self, tmp_path: Path):
        out = tmp_path / "nested" / "file.json"
        payload = {"a": 1, "b": [1, 2, 3]}
        safe_write_json(out, payload)
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data == payload

    def test_safe_write_json_raises_oserror(self, monkeypatch, tmp_path: Path):
        out = tmp_path / "boom.json"

        def boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(builtins, "open", boom)
        with pytest.raises(OSError):
            safe_write_json(out, {"x": 1})
