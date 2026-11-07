"""Tiny G-code writer with Fanuc number formatting."""
from __future__ import annotations


def f(v: float, nd: int = 3) -> str:
    """Fanuc-safe number: always a decimal point (X10 means 10 microns on many controls)."""
    s = f"{v:.{nd}f}".rstrip("0")
    if s.endswith("."):
        return s
    if "." not in s:
        s += "."
    return "0." if s in ("-0.", "-.") else s


class GCode:
    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, *parts):
        self.lines.append(" ".join(str(p) for p in parts if p not in (None, "")))

    def comment(self, text: str):
        # Fanuc comments cannot contain ( or ); keep them uppercase and short
        import textwrap
        t = text.replace("(", "[").replace(")", "]").upper()
        for part in textwrap.wrap(t, 58) or [""]:
            self.lines.append(f"({part})")

    def blank(self):
        self.lines.append("")

    def drop_unused_tool_comments(self, used: set):
        """Header lists every turret tool; keep only the ones the program calls."""
        import re
        out = []
        for ln in self.lines:
            m = re.match(r"\(T(\d+) ", ln)
            if m and int(m.group(1)) not in used and not ln.startswith("(T" + m.group(1) + " M06"):
                continue
            out.append(ln)
        self.lines = out

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"
