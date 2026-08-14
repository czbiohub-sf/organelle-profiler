"""Central base path for organelle_profiler storage roots.

All data/config/analysis locations derive from ``BASE_PATH``, which must be
supplied via the ``OPS_BASE_PATH`` environment variable — there is no default,
so the package never silently reads or writes somebody else's storage:

    export OPS_BASE_PATH=/path/to/ops_data
"""
import os


def _require(var: str) -> str:
    """Return env var ``var``, or raise with a usable message if it is unset."""
    value = os.environ.get(var)
    if not value:
        raise RuntimeError(
            f"{var} is not set. Point it at your storage root, e.g. "
            f"`export {var}=/path/to/ops_data`."
        )
    return value


BASE_PATH = _require("OPS_BASE_PATH")
