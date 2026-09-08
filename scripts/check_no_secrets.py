"""Simple repository secret scanner used by CI as a lightweight guardrail."""

from __future__ import annotations

import re
import sys
from pathlib import Path


# Keep patterns intentionally simple and conservative for CI use.
SECRET_PATTERNS = {
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "github_token": re.compile(r"ghp_[A-Za-z0-9]{36}"),
    "generic_api_key": re.compile(r"(?i)api[_-]?key\s*[=:]\s*[\"\'][^\"\']+[\"\']"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
}

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "env", "models", "results"}
SKIP_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".png", ".jpg", ".jpeg", ".pdf"}


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.parts) or path.suffix.lower() in SKIP_SUFFIXES


def scan_repo(root: Path) -> list[str]:
    findings: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or should_skip(path):
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                findings.append(f"{path}: matched pattern '{name}'")
    return findings


def main() -> int:
    root = Path.cwd()
    findings = scan_repo(root)
    if findings:
        # Do not print possible secret values or related file details to logs.
        print(f"Potential secrets found: {len(findings)} possible match(es).")
        return 1

    print("No obvious secrets detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
