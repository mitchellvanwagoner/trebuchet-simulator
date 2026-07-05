"""Array backend selection: numpy by default, optional CuPy GPU acceleration.

Set the TREBUCHET_GPU=1 environment variable to try CuPy before falling back
to numpy.
"""

import os

_want_gpu = os.environ.get("TREBUCHET_GPU", "0") == "1"

GPU_AVAILABLE = False

if _want_gpu:
    try:
        import cupy as np

        np.array([1.0]).sum()  # verify the runtime actually works
        GPU_AVAILABLE = True
    except Exception:
        import numpy as np
else:
    import numpy as np


def to_cpu(array):
    """Convert a GPU array to CPU; no-op when running on numpy."""
    if GPU_AVAILABLE and hasattr(array, "get"):
        return array.get()
    return array
