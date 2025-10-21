import ntpath


def path_leaf(path: str) -> str:
    """Return the final component of ``path`` regardless of trailing separators."""

    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)
