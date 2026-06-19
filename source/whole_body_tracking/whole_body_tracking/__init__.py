"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
try:
    from .tasks import *
except ModuleNotFoundError as exc:
    missing_name = exc.name or ""
    if missing_name not in {"isaaclab_tasks", "isaaclab"} and not missing_name.startswith("omni"):
        raise
