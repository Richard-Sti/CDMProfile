# Copyright (C) 2023 Richard Stiskalek, Deaglan Bartlett
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
Generate functions of a given complexity for a run.
"""
import os
from argparse import ArgumentParser
from os import listdir, makedirs
from os.path import abspath, dirname, exists, isfile, join
from shutil import move

import esr.generation.duplicate_checker  # noqa
import esr.generation.generator as generator  # noqa
from mpi4py import MPI

from utils import read_config


def generate_functions(runname, comp):
    """Generate functions of a given complexity for a run."""
    esr.generation.duplicate_checker.main(runname, comp)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--runname", type=str,
                        help="ESR run name, defining the basis functions.")
    parser.add_argument("--comp", type=int, help="Function complexity.")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    # Run the generator
    generate_functions(args.runname, args.comp)

    # Move the generated functions to the CDMProfile directory
    if rank == 0:
        srcdir = abspath(
            join(dirname(generator.__file__), '..', 'function_library'))
        srcdir = join(srcdir, args.runname, f"compl_{args.comp}")

        config = read_config()
        targetdir = join(config["path"]["results"], args.runname,
                         f"compl_{args.comp}")
        if not exists(targetdir):
            makedirs(targetdir)

        print(f"Moving functions from `{srcdir}` to `{targetdir}`")
        for filename in listdir(srcdir):
            file_path = join(srcdir, filename)
            if isfile(file_path):
                move(file_path, os.path.join(targetdir, filename))
