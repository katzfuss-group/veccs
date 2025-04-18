import numpy as np
from deprecated import deprecated


def has_legacy_support() -> bool:
    """
    Check if the legacy support is available.
    """
    try:
        import faiss  # noqa: F401
        import sklearn  # noqa: F401

        return True
    except ImportError:
        return False


try:
    from ._faiss import find_nns_l2
except ImportError:

    @deprecated
    def find_nns_l2(locs: np.ndarray, max_nn: int = 10) -> np.ndarray:
        raise ImportError(
            "`faiss` is not installed. Please install it with `[legacy]` to use this "
            "feature."
        )


try:
    from ._sklearn import find_nns_naive

except ImportError:

    @deprecated
    def find_nns_naive(
        locs: np.ndarray, dist_fun: str = "euclidean", max_nn: int = 10, **kwargs
    ) -> np.ndarray:
        raise ImportError(
            "`sklearn` is not installed. Please install it with `[legacy]` to use this "
            "feature."
        )


__all__ = [
    "find_nns_l2",
    "find_nns_naive",
    "has_legacy_support",
]
