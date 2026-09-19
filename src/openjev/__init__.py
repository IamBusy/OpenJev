"""OpenJev: independent typed-decision experiments."""

__version__ = "0.3.2"

__all__ = ["OpenJevModel", "__version__"]


def __getattr__(name):
    # Keep the historical MiniLM installation usable without the Qwen extra.
    if name == "OpenJevModel":
        from .branch_model import BranchDecision

        return BranchDecision
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
