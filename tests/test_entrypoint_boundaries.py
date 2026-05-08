from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _python_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if path.is_file() and "graphify-out" not in path.parts and ".venv" not in path.parts
    )


def test_non_legacy_modules_do_not_import_default_universe_from_main() -> None:
    offenders: list[str] = []
    allowed = {
        ROOT / "main.py",
    }
    for path in _python_files(ROOT):
        if path in allowed or path.parts[-2:] == ("tests", "test_entrypoint_boundaries.py"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "main":
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} from main import ...")
    assert not offenders, "Legacy main.py imports remain in active code paths:\n" + "\n".join(offenders)


def test_root_qc_modules_remain_thin_facades() -> None:
    for rel in ["qc_main.py", "qc_alpha_model.py"]:
        path = ROOT / rel
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        class_defs = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        func_defs = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        assert not class_defs, f"{rel} should remain a facade, not define classes"
        assert not func_defs, f"{rel} should remain a facade, not define functions"
