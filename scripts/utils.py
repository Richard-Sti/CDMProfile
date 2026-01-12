# Copyright (C) 2024 Richard Stiskalek
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
"""
Utility functions for the CDM profile analysis.
"""
import tomllib
from os.path import dirname, exists, join

PROJECT_DIR = join(dirname(__file__), "..")
CONFIG_PATH = join(PROJECT_DIR, "config.toml")
LOCAL_CONFIG_PATH = join(PROJECT_DIR, "local_config.toml")

LOCAL_CONFIG_TEMPLATE = """
[path]
results = "/path/to/results"
data = "/path/to/data"
venv = "/path/to/venv"
""".strip()


def read_config():
    """
    Read the configuration file, merging with local_config.toml.

    Returns
    -------
    dict
    """
    if not exists(LOCAL_CONFIG_PATH):
        raise FileNotFoundError(
            f"Local config file not found: {LOCAL_CONFIG_PATH}\n"
            f"Please create it with the following contents:\n\n"
            f"{LOCAL_CONFIG_TEMPLATE}"
        )

    with open(CONFIG_PATH, "rb") as f:
        config = tomllib.load(f)

    with open(LOCAL_CONFIG_PATH, "rb") as f:
        local_config = tomllib.load(f)

    # Merge local_config into config (local takes precedence)
    for key, value in local_config.items():
        if key in config and isinstance(config[key], dict):
            config[key].update(value)
        else:
            config[key] = value

    return config
