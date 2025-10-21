"""Active utility helpers."""
from __future__ import annotations

import ntpath
import os
from typing import Union

PathLike = Union[str, os.PathLike[str]]


def path_leaf(path: PathLike) -> str:
    """Return the final component of ``path`` regardless of trailing separators."""
    head, tail = ntpath.split(str(path))
    return tail or ntpath.basename(head)
