from astropy.io import fits
import healpy as hp
import numpy as np


def apwnorm(x, min=None, max=None):
    if min is None:
        min = np.nanmin(x[~np.isinf(x)])
    
    if max is None:
        max = np.nanmax(x[~np.isinf(x)])
        
    return np.clip((x - min) / (max - min), 0, 1)


# ------------------------------------------------
# Projections:
projs = dict()

# Mollweide ICRS:
projs['icrs'] = hp.projector.MollweideProj(xsize=1024, rot=[180, 0])

# Mollweide Galacticanticenter:
projs['gal-180'] = hp.projector.MollweideProj(xsize=1024, 
                                             rot=[86.40498829, 28.93617776, -59.])

# Mollweide Galactic:
projs['gal'] = hp.projector.MollweideProj(xsize=1024, 
                                          rot=[266.40498829, -28.93617776, 59.])

# Mollweide Sagittarius:
projs['sag'] = hp.projector.MollweideProj(xsize=1024, 
                                          rot=[284.03876751, -29.00408353, -10.])

projs['gd1'] = hp.projector.MollweideProj(xsize=1024, 
                                          rot=[200., 59.4504341, 13.])


# Orthographic
projs['icrs-ortho'] = hp.projector.OrthographicProj(xsize=1024, 
                                                    rot=[180, 45, 0])


def get_data(bass_file=None, decals_file=None):
    if bass_file is None and decals_file is None:
        raise ValueError()
    
    if bass_file is not None:
        bass_cube = fits.getdata(bass_file, 0) / 1.5  # TODO: MAGIC NUMBER
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
        bass_cube = np.zeros_lile(decals_cube)
        
    tmp_bass = bass_cube.copy()
    tmp_bass[decals_cube != 0] = 0.

    both = decals_cube + tmp_bass

    cubesum = np.sum(both, axis=1) / nslice
    footprint_mask = cubesum != 0
    
    return both, cube_dm, footprint_mask