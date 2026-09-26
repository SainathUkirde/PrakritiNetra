# Data layer package
#
# SYNTHETIC DATA (fallback — no real files needed):
# from .synthetic import load_dataset          # noqa: F401

# REAL GFS-only (single source):
# from .gfs_loader import load_dataset         # noqa: F401

# REAL MULTI-SOURCE — GFS (nwp) + GEFS (ensemble) + ERA5 (truth): ACTIVE
from .real_loader import load_dataset        # noqa: F401

__all__ = ["load_dataset"]
