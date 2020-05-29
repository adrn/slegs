from astropy.io import fits
import numpy as np


def get_data(bass_file, decals_file, stitch=False):

    cubes = {}

    cubes['bass'] = fits.getdata(bass_file, 0)
    distmods_bass = fits.getdata(bass_file, 1)

    cubes['decals'] = fits.getdata(decals_file, 0)
    distmods_decals = fits.getdata(decals_file, 1)

    if not np.allclose(distmods_bass, distmods_decals):
        raise ValueError("distmods don't match")

    elif cubes['bass'].shape != cubes['decals'].shape:
        raise ValueError("shapes don't match")

    npix, nslice = cubes['bass'].shape

    footprints = {}
    for k in cubes.keys():
        cubesum = np.sum(cubes[k], axis=1) / nslice
        footprints[k] = cubesum != 0

    if stitch:
        stitched = get_all_stitched(cubes, footprints)
        footprint = footprints['bass'] | footprints['decals']
        return stitched, distmods_decals, footprint

    else:
        return cubes, distmods_decals, footprints


def stitch_slice(bass, decals, overlap, bass_footprint,
                 min_bass=0, min_decals=0):
    mask = (bass > min_bass) & (decals > min_decals) & overlap
    poly = np.poly1d(np.polyfit(bass[mask], decals[mask],
                                deg=1))

    overlap = overlap & (bass > 0) & (decals > 0)

    new_bass = poly(bass)
    stitched = decals.copy()
    stitched[bass_footprint] += new_bass[bass_footprint]
    stitched[overlap] *= 0.5

    return stitched


def get_all_stitched(cubes, footprints):
    overlap = (footprints['bass'] * footprints['decals']).astype(bool)

    npix, nslice = cubes['bass'].shape

    all_stitched = []
    for i in range(nslice):
        stitched = stitch_slice(cubes['bass'][:, i], cubes['decals'][:, i],
                                overlap, footprints['bass'],
                                min_bass=4)
        all_stitched.append(stitched)
    all_stitched = np.stack(all_stitched, axis=1)

    return all_stitched


def apwnorm(x, min=None, max=None):
    if min is None:
        min = np.nanmin(x[~np.isinf(x)])
    if max is None:
        max = np.nanmax(x[~np.isinf(x)])
    return np.clip((x - min) / (max - min), 0, 1)