"""Diagnostics collected while converting."""

from dataclasses import dataclass, field, asdict
import sys

ERROR, WARNING, NOTE = "error", "warning", "note"


@dataclass
class Diagnostic:
    level: str
    message: str
    file: str = ""
    line: int = 0
    hint: str = ""

    def format(self, color=False):
        loc = f"{self.file}:{self.line}: " if self.file else ""
        lvl = self.level
        if color:
            c = {"error": "31", "warning": "33", "note": "36"}[self.level]
            lvl = f"\033[1;{c}m{lvl}\033[0m"
        s = f"{loc}{lvl}: {self.message}"
        if self.hint:
            s += f"\n    hint: {self.hint}"
        return s


@dataclass
class Diagnostics:
    items: list = field(default_factory=list)
    _seen: set = field(default_factory=set)

    def add(self, level, message, file="", line=0, hint=""):
        key = (level, message, file, line)
        if key in self._seen:
            return
        self._seen.add(key)
        self.items.append(Diagnostic(level, message, file, line, hint))

    def error(self, *a, **k):
        self.add(ERROR, *a, **k)

    def warning(self, *a, **k):
        self.add(WARNING, *a, **k)

    def note(self, *a, **k):
        self.add(NOTE, *a, **k)

    @property
    def errors(self):
        return [d for d in self.items if d.level == ERROR]

    def dump(self, stream=None):
        stream = stream or sys.stderr
        color = hasattr(stream, "isatty") and stream.isatty()
        for d in self.items:
            print(d.format(color), file=stream)

    def to_list(self):
        return [asdict(d) for d in self.items]
