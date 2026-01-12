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
Utility functions for CDM profile analysis.
"""
from pathlib import Path

import h5py
import numpy as np


class HaloData:
    """
    Container for halo particle radii with offset-based indexing.

    Parameters
    ----------
    radii : ndarray
        Concatenated particle radii (N_total,).
    halo_ids : ndarray
        Halo IDs in order (N_halos,).
    offset : ndarray
        Start/end indices into radii array (N_halos+1,).
    """

    def __init__(self, radii, halo_ids, offset):
        self.radii = radii
        self.halo_ids = halo_ids
        self.offset = offset
        self._id_to_idx = {hid: i for i, hid in enumerate(halo_ids)}

    @property
    def n_halos(self):
        """Return number of halos."""
        return len(self.halo_ids)

    def load(self, halo_id):
        """Load radii for a single halo by ID."""
        if halo_id not in self._id_to_idx:
            raise ValueError(f"Halo {halo_id} not found")
        idx = self._id_to_idx[halo_id]
        return self.radii[self.offset[idx]:self.offset[idx + 1]]

    def load_by_index(self, idx):
        """Load radii for a halo by index (0 to n_halos-1)."""
        return self.radii[self.offset[idx]:self.offset[idx + 1]]

    def __iter__(self):
        """Iterate over (halo_id, radii) pairs."""
        for i, hid in enumerate(self.halo_ids):
            yield hid, self.radii[self.offset[i]:self.offset[i + 1]]

    def __len__(self):
        return self.n_halos

    def __contains__(self, halo_id):
        return halo_id in self._id_to_idx

    def __getitem__(self, halo_id):
        """Get radii by halo ID."""
        return self.load(halo_id)

    def bin(self, nbin, log=True, indices=None):
        """
        Bin particle radii into counts. Each halo is binned between its
        own rmin and rmax.

        Parameters
        ----------
        nbin : int
            Number of bins.
        log : bool, optional
            If True, use logarithmic bins. Default: True.
        indices : array-like, optional
            Indices of halos to bin (0 to n_halos-1). Default: all halos.

        Returns
        -------
        dict with keys:
            bin_counts : (nhalo, nbin) int
            bin_positions : (nhalo, nbin) float - bin centers per halo
            bin_edges : (nhalo, nbin+1) float - bin edges per halo
            rmin : (nhalo,) float - per halo
            rmax : (nhalo,) float - per halo
            halo_ids : (nhalo,) int
            npart : (nhalo,) int
        """
        if indices is None:
            indices = np.arange(self.n_halos)
        indices = np.asarray(indices)

        nhalo = len(indices)
        bin_counts = np.zeros((nhalo, nbin), dtype=np.int64)
        bin_positions = np.zeros((nhalo, nbin), dtype=np.float64)
        bin_edges = np.zeros((nhalo, nbin + 1), dtype=np.float64)
        rmin_arr = np.zeros(nhalo, dtype=np.float64)
        rmax_arr = np.zeros(nhalo, dtype=np.float64)
        npart = np.zeros(nhalo, dtype=np.int64)

        for i, idx in enumerate(indices):
            radii = self.load_by_index(idx)
            rmin = radii.min()
            rmax = radii.max()
            rmin_arr[i] = rmin
            rmax_arr[i] = rmax

            if log:
                edges = np.logspace(np.log10(rmin), np.log10(rmax), nbin + 1)
                positions = np.sqrt(edges[:-1] * edges[1:])
            else:
                edges = np.linspace(rmin, rmax, nbin + 1)
                positions = 0.5 * (edges[:-1] + edges[1:])

            bin_edges[i] = edges
            bin_positions[i] = positions
            bin_counts[i], _ = np.histogram(radii, bins=edges)
            npart[i] = len(radii)

        return {
            'bin_counts': bin_counts,
            'bin_positions': bin_positions,
            'bin_edges': bin_edges,
            'rmin': rmin_arr,
            'rmax': rmax_arr,
            'halo_ids': self.halo_ids[indices],
            'npart': npart,
        }


def load_from_folder(directory, verbose=True):
    """
    Load halo radii from a folder of halo_*.npy files.

    Parameters
    ----------
    directory : str or Path
        Path to directory containing halo_*.npy files.
    verbose : bool, optional
        If True, show progress bar. Default: True.

    Returns
    -------
    HaloData
    """
    from tqdm import tqdm

    directory = Path(directory)

    # Find all halo files and sort by ID
    halo_files = {}
    for f in directory.glob("halo_*.npy"):
        halo_id = int(f.stem.split("_")[1])
        halo_files[halo_id] = f

    halo_ids = np.array(sorted(halo_files.keys()))

    # Load and concatenate
    radii_list = []
    offset = [0]
    iterator = tqdm(halo_ids, desc="Loading halos", disable=not verbose)
    for hid in iterator:
        r = np.load(halo_files[hid])
        radii_list.append(r)
        offset.append(offset[-1] + len(r))

    radii = np.concatenate(radii_list)
    offset = np.array(offset)

    return HaloData(radii, halo_ids, offset)


def load_from_hdf5(filepath):
    """
    Load halo radii from HDF5 file.

    Parameters
    ----------
    filepath : str or Path
        Path to HDF5 file with radii/halo_id/offset structure.

    Returns
    -------
    HaloData
    """
    with h5py.File(filepath, 'r') as f:
        radii = f['radii'][:]
        halo_ids = f['halo_id'][:]
        offset = f['offset'][:]
    return HaloData(radii, halo_ids, offset)


def save_binned_halos(filepath, binned_data):
    """
    Save binned halo data to HDF5 file.

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    binned_data : dict
        Output from bin_halos().
    """
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('bin_counts', data=binned_data['bin_counts'])
        f.create_dataset('bin_positions', data=binned_data['bin_positions'])
        f.create_dataset('bin_edges', data=binned_data['bin_edges'])
        f.create_dataset('halo_ids', data=binned_data['halo_ids'])
        f.create_dataset('npart', data=binned_data['npart'])
        f.attrs['rmin'] = binned_data['rmin']
        f.attrs['rmax'] = binned_data['rmax']


def load_binned_halos(filepath):
    """
    Load binned halo data from HDF5 file.

    Parameters
    ----------
    filepath : str or Path
        Input file path.

    Returns
    -------
    dict
        Same structure as bin_halos() output.
    """
    with h5py.File(filepath, 'r') as f:
        return {
            'bin_counts': f['bin_counts'][:],
            'bin_positions': f['bin_positions'][:],
            'bin_edges': f['bin_edges'][:],
            'halo_ids': f['halo_ids'][:],
            'npart': f['npart'][:],
            'rmin': f.attrs['rmin'],
            'rmax': f.attrs['rmax'],
        }
