import astropy.coordinates as coord
import astropy.units as u
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

def obj_func(ang, icrs, R):
    R3 = rotation_matrix(ang, 'x')
    rep = icrs.cartesian.transform(R3 @ R).represent_as(
        coord.UnitSphericalRepresentation)
    return np.sum(rep.lat.degree**2)

def get_roll(icrs, R, x0=0.):
    res = minimize(obj_func, x0=x0, args=(icrs, R))
    return coord.Angle(res.x[0] * u.deg)

def get_origin(fr):
    usph_origin = coord.UnitSphericalRepresentation(0*u.deg, 0*u.deg)
    return fr.realize_frame(usph_origin).transform_to(coord.ICRS)

def get_rot(fr, x0=0.):
    origin = get_origin(fr)
    usph_band  = coord.UnitSphericalRepresentation(np.random.uniform(0, 360, 10000)*u.deg, 
                                                   np.random.uniform(-1, 1, size=10000)*u.deg)
    band_icrs = fr.realize_frame(usph_band).transform_to(coord.ICRS)
    
    R1 = rotation_matrix(origin.ra, 'z')
    R2 = rotation_matrix(-origin.dec, 'y')
    Rtmp = R2 @ R1
    
    roll = get_roll(band_icrs, Rtmp, x0)
    
    return [origin.ra.degree, origin.dec.degree, roll.degree]

from astropy.coordinates.matrix_utilities import rotation_matrix
from scipy.optimize import minimize
import gala.coordinates as gc
projs['orp'] = hp.projector.MollweideProj(xsize=1024, 
                                          rot=get_rot(gc.Orphan()))
projs['pal5'] = hp.projector.MollweideProj(xsize=1024, 
                                           rot=get_rot(gc.Pal5()))


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
        bass_cube = np.zeros_lile(decals_cube)
        
    tmp_bass = bass_cube.copy()
    tmp_bass[decals_cube != 0] = 0.

    both = decals_cube + tmp_bass

    cubesum = np.sum(both, axis=1) / nslice
    footprint_mask = cubesum != 0
    
    return both, cube_dm, footprint_mask