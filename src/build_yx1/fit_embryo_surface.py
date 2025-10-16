"""Legacy sphere-fitting module (deprecated)."""
from warnings import warn

from src.geometry.sphere import (
    fit_sphere,
    fit_sphere_with_percentile,
    fit_spheres_for_well,
    make_sphere_mesh,
    sphere_fit_wrapper,
)

warn(
    "src.build_yx1.fit_embryo_surface is deprecated; import from src.geometry instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "fit_sphere",
    "fit_sphere_with_percentile",
    "fit_spheres_for_well",
    "make_sphere_mesh",
    "sphere_fit_wrapper",
]
