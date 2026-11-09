"""Editable agent instructions shipped with the MAS package."""

from importlib.resources import files


def load_prompt(name):
    """Read the current instruction text for each request, without caching."""
    return files(__package__).joinpath(f"{name}.txt").read_text(encoding="utf-8")
