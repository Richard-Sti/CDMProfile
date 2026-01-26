# Copyright (C) 2026 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""CDM Profile package."""

from . import asymptotes, data, fitting, symbolic
from .asymptotes import compute_asymptotes_with_params
from .data import HaloData, load_from_folder
from .fitting import compile_fitter, compile_nested_fitter, fit_nfw
from .symbolic import SympyParser

__version__ = "0.1.0"

__all__ = [
    "asymptotes",
    "data",
    "fitting",
    "symbolic",
    "HaloData",
    "load_from_folder",
    "compile_fitter",
    "compile_nested_fitter",
    "fit_nfw",
    "compute_asymptotes_with_params",
    "SympyParser",
]
