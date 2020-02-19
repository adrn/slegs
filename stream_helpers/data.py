from astropy.io import fits
import numpy as np


def get_data(bass_file=None, decals_file=None, bass_scale_factor=1.5):
    if bass_file is None and decals_file is None:
        raise ValueError()

    if bass_file is not None:
        bass_cube = fits.getdata(bass_file, 0) / bass_scale_factor
        cube_dm = fits.getdata(bass_file, 1)
        npix, nslice = bass_cube.shape
    else:
        bass_cube = None

    if decals_file is not None:
        decals_cube = fits.getdata(decals_file, 0)
        cube_dm = fits.getdata(decals_file, 1)
        npix, nslice = decals_cube.shape
    else:
        decals_cube = None

    if decals_cube is None:
        decals_cube = np.zeros_like(bass_cube)
    elif bass_cube is None:
        bass_cube = np.zeros_like(decals_cube)

    tmp_bass = bass_cube.copy()
    tmp_bass[decals_cube != 0] = 0.

    both = decals_cube + tmp_bass

    cubesum = np.sum(both, axis=1) / nslice
    footprint_mask = cubesum != 0

    return both, cube_dm, footprint_mask
