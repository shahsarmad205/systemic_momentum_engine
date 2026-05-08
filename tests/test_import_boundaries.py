from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_IMPORT_ROOTS = {"analysis", "research"}
LIVE_BOUNDARY_PATHS = [
    ROOT / "brokers",
    ROOT / "risk",
    ROOT / "run_daily_pipeline.py",
    ROOT / "run_live_trading.py",
    ROOT / "qc_main.py",
    ROOT / "qc_alpha_model.py",
    ROOT / "LeanCloud" / "BinaryEdge",
]


def _python_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.py") if p.is_file())


def _forbidden_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", maxsplit=1)[0]
                if root in FORBIDDEN_IMPORT_ROOTS:
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno} import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", maxsplit=1)[0]
            if root in FORBIDDEN_IMPORT_ROOTS:
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno} from {node.module} import ...")
    return violations


def test_live_code_does_not_import_research_or_analysis_modules() -> None:
    violations: list[str] = []
    for boundary_path in LIVE_BOUNDARY_PATHS:
        for py_file in _python_files(boundary_path):
            violations.extend(_forbidden_imports(py_file))

    assert not violations, "Research/analysis imports crossed live boundary:\n" + "\n".join(violations)
