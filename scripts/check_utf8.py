#!/usr/bin/env python3
"""Verify files are valid UTF-8 with no U+FFFD / cp1252 damage / stray '?' runs typical of encoding loss. Usage: check_utf8.py FILE..."""
import re
import sys

bad = 0
for p in sys.argv[1:]:
    data = open(p, "rb").read()
    try:
        s = data.decode("utf-8")
    except UnicodeDecodeError as e:
        print(f"BAD-UTF8 {p}: {e}")
        bad += 1
        continue
    issues = []
    if "\ufffd" in s:
        issues.append("U+FFFD")
    if re.search(r"[\u4e00-\u9fff]\?[\u4e00-\u9fff]", s):
        issues.append("?-in-CJK (lost char)")
    if re.search(r"\?\?\?", s):
        issues.append("???-run")
    cjk = len(re.findall(r"[\u4e00-\u9fff]", s))
    print(f"{'WARN' if issues else 'OK  '} {p}: {len(s)} chars, {cjk} CJK {' '.join(issues)}")
    bad += bool(issues)
sys.exit(1 if bad else 0)
