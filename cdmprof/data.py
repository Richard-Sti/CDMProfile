# Copyright (C) 2025 Richard Stiskalek
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

import numpy as np
from h5py import File
from tqdm import tqdm


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

    def bin(self, nbin, log=True, indices=None, n_sample=None, seed=None,
            verbose=True):
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
        n_sample : int, optional
            If specified, randomly downsample each halo to this many particles
            before binning. Halos with fewer particles are kept as-is.
        seed : int, optional
            Random seed for reproducible downsampling.
        verbose : bool, optional
            If True, show progress bar. Default: True.

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
        rmin_arr = np.zeros(nhalo, dtype=np.float64)
        rmax_arr = np.zeros(nhalo, dtype=np.float64)
        npart = np.zeros(nhalo, dtype=np.int64)

        # Setup random generator for downsampling
        rng = np.random.default_rng(seed) if n_sample is not None else None
        n_upsampled = 0  # Track halos that needed upsampling

        # Compute bin counts using direct indexing (faster than np.histogram)
        iterator = tqdm(enumerate(indices), total=nhalo, desc="Binning",
                        disable=not verbose)
        for i, idx in iterator:
            radii = self.load_by_index(idx)

            # Resample if requested
            if n_sample is not None:
                if len(radii) > n_sample:
                    # Downsample without replacement
                    radii = rng.choice(radii, size=n_sample, replace=False)
                elif len(radii) < n_sample:
                    # Upsample with replacement
                    radii = rng.choice(radii, size=n_sample, replace=True)
                    n_upsampled += 1

            rmin = radii.min()
            rmax = radii.max()
            rmin_arr[i] = rmin
            rmax_arr[i] = rmax
            npart[i] = len(radii)

            # Direct bin index computation
            if log:
                log_min = np.log(rmin)
                log_range = np.log(rmax) - log_min
                bin_idx = ((np.log(radii) - log_min) * (nbin / log_range))
            else:
                bin_idx = (radii - rmin) * (nbin / (rmax - rmin))

            bin_idx = np.clip(bin_idx.astype(np.int64), 0, nbin - 1)
            bin_counts[i] = np.bincount(bin_idx, minlength=nbin)[:nbin]

        # Vectorized computation of edges and positions
        t = np.linspace(0, 1, nbin + 1)
        if log:
            log_rmin = np.log(rmin_arr)[:, None]
            log_rmax = np.log(rmax_arr)[:, None]
            bin_edges = np.exp(log_rmin + t * (log_rmax - log_rmin))
            bin_positions = np.sqrt(bin_edges[:, :-1] * bin_edges[:, 1:])
        else:
            bin_edges = rmin_arr[:, None] + t * (rmax_arr - rmin_arr)[:, None]
            bin_positions = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])

        # Ensure exact boundary values (avoid floating-point drift)
        bin_edges[:, 0] = rmin_arr
        bin_edges[:, -1] = rmax_arr

        # Warn if any halos were upsampled
        if n_upsampled > 0:
            print(f"Warning: {n_upsampled}/{nhalo} halos had fewer than "
                  f"{n_sample} particles and were upsampled with replacement")

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
    with File(filepath, 'r') as f:
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
    with File(filepath, 'w') as f:
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
    with File(filepath, 'r') as f:
        return {
            'bin_counts': f['bin_counts'][:],
            'bin_positions': f['bin_positions'][:],
            'bin_edges': f['bin_edges'][:],
            'halo_ids': f['halo_ids'][:],
            'npart': f['npart'][:],
            'rmin': f.attrs['rmin'],
            'rmax': f.attrs['rmax'],
        }
