"""Legacy spherical harmonics helpers (deprecated)."""
from warnings import warn

from src.geometry.spherical_harmonics import (
    build_sh_basis,
    cart2sph,
    create_sh_mesh,
    fit_sphere_and_sh,
    sph2cart,
)
from src.geometry.sphere import create_sphere_mesh

warn(
    "src.build_lightsheet.fit_embryo_surface is deprecated; import from src.geometry instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "build_sh_basis",
    "cart2sph",
    "create_sh_mesh",
    "create_sphere_mesh",
    "fit_sphere_and_sh",
    "sph2cart",
]
