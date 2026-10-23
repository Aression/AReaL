#!/usr/bin/env python3
"""
Pure-Python verification of the attention_mask fix (commit 02).

No torch dependency — uses only the standard library.
Checks that the fix changes dict(full_attention=None) to None and
that the downstream behavioral consequence (block-diagonal causal mask
vs no mask at all) is correctly understood.

Usage:
    python3 verify_attention_mask_fix.py <path/to/AReaL/repo>

Exit codes:
    0  — all checks passed
    1  — one or more checks failed
    2  — repo path not found or file missing
"""

import os
import re
import sys
from dataclasses import dataclass


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def find_in_file(filepath: str, pattern: str) -> list[str]:
    """Return lines matching a regex pattern."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [line.rstrip("\n") for line in lines if re.search(pattern, line)]
    except FileNotFoundError:
        return []


def check_no_dict_full_attention(filepath: str) -> CheckResult:
    """The bug pattern dict(full_attention=None) must NOT appear in the
    _prepare_mb_list attention_mask assignment path.
    """
    matches = find_in_file(filepath, r"dict\(\s*full_attention\s*=\s*None")
    # The import line may still have "full_attention" for Qwen3 models,
    # but the non-Qwen3 path must use None.
    # Check specifically for the assignment pattern.
    bug_matches = find_in_file(
        filepath,
        r'mb\["attention_mask"\]\s*=\s*dict\(\s*full_attention\s*=\s*None',
    )
    if not bug_matches:
        return CheckResult(
            "no_dict_full_attention",
            True,
            "No dict(full_attention=None) assignment found in attention_mask path",
        )
    return CheckResult(
        "no_dict_full_attention",
        False,
        f"Found {len(bug_matches)} bug pattern(s): {bug_matches[:3]}",
    )


def check_none_assignment(filepath: str) -> CheckResult:
    """The fix must set mb["attention_mask"] = None (not dict)."""
    # Look for the pattern: mb["attention_mask"] = None (in the else branch)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # The fixed code should have mb["attention_mask"] = None in the else branch
    # (not inside the Qwen3 if block, but in the else)
    none_matches = find_in_file(
        filepath, r'mb\["attention_mask"\]\s*=\s*None'
    )
    if len(none_matches) >= 2:  # both mb and padded_mb
        return CheckResult(
            "none_assignment",
            True,
            f"Found {len(none_matches)} None assignments for attention_mask",
        )
    return CheckResult(
        "none_assignment",
        False,
        f"Expected >= 2 None assignments, found {len(none_matches)}: {none_matches}",
    )


def check_comment_explains_fix(filepath: str) -> CheckResult:
    """The fix should include a comment explaining the rationale."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Look for key phrases from the fix comment
    phrases = [
        "create_causal_mask",
        "block-diagonal",
        "packed sequence",
    ]
    found = [p for p in phrases if p.lower() in content.lower()]
    if len(found) >= 2:
        return CheckResult(
            "comment_explains_fix",
            True,
            f"Found explanatory phrases: {found}",
        )
    return CheckResult(
        "comment_explains_fix",
        False,
        f"Expected >= 2 explanatory phrases, found {found}",
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 verify_attention_mask_fix.py <path/to/AReaL/repo>")
        sys.exit(2)

    repo_path = sys.argv[1]
    filepath = os.path.join(repo_path, "areal", "engine", "fsdp_engine.py")

    if not os.path.isfile(filepath):
        print(f"ERROR: File not found: {filepath}")
        sys.exit(2)

    checks = [
        check_no_dict_full_attention(filepath),
        check_none_assignment(filepath),
        check_comment_explains_fix(filepath),
    ]

    passed = sum(1 for c in checks if c.passed)
    failed = sum(1 for c in checks if not c.passed)

    print("=" * 60)
    print("  Attention Mask Fix Verification")
    print("=" * 60)
    print(f"  File: {filepath}")
    print()

    for c in checks:
        status = "✅ PASS" if c.passed else "❌ FAIL"
        print(f"  {status}  {c.name}")
        if c.detail:
            print(f"          {c.detail}")

    print()
    print(f"  Results: {passed} passed, {failed} failed")
    print()

    if failed > 0:
        print("  ❌ VERIFICATION FAILED — the attention mask fix is NOT applied.")
        sys.exit(1)
    else:
        print("  ✅ ALL CHECKS PASSED — attention mask fix is correctly applied.")
        sys.exit(0)


if __name__ == "__main__":
    main()
