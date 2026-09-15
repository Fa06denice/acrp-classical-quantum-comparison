"""Instance input/output: AMPL ``.dat`` parsing and zip-aware loading."""

from .ampl_dat import AmplData, parse_ampl_dat, serialize_ampl_dat
from .loader import InstanceLoader, discover_instances

__all__ = [
    "AmplData",
    "parse_ampl_dat",
    "serialize_ampl_dat",
    "InstanceLoader",
    "discover_instances",
]
