import astropy.coordinates as coord
import astropy.units as u
import gala.coordinates as gc
import numpy as np
from scipy.optimize import minimize

__all__ = ['zyx_euler_from_endpoints']


def zyx_euler_from_endpoints(lon1, lat1, lon2, lat2):
    c1 = coord.SkyCoord(lon1*u.deg, lat1*u.deg)
    c2 = coord.SkyCoord(lon2*u.deg, lat2*u.deg)
    fr = gc.GreatCircleICRSFrame.from_endpoints(c1, c2)
    origin = fr.realize_frame(coord.UnitSphericalRepresentation(0*u.deg,
                                                                0*u.deg))

    gc_icrs = origin.transform_to(coord.ICRS)
    R = gc.greatcircle.reference_to_greatcircle(coord.ICRS, fr)
    psi = -np.degrees(np.arctan2(R[2, 1], R[2, 2]))

    return [gc_icrs.ra.degree, gc_icrs.dec.degree, psi]


def obj_func(ang, icrs, R):
    R3 = coord.matrix_utilities.rotation_matrix(ang, 'x')
    rep = icrs.cartesian.transform(R3 @ R).represent_as(
        coord.UnitSphericalRepresentation)
    return np.sum(rep.lat.degree**2)


def get_roll(icrs, R, x0=0.):
    res = minimize(obj_func, x0=x0, args=(icrs, R))
    return coord.Angle(res.x[0] * u.deg)


def get_rot(frame, x0=0.):

    trans = coord.frame_transform_graph.get_transform(coord.ICRS,
                                                      frame.__class__)
    for t in trans.transforms:
        if not isinstance(t, coord.transformations.StaticMatrixTransform):
            R = None
            break

    else:
        # All are static matrix transformations
        R = np.eye(3)
        for t in trans.transforms:
            R = R @ t.matrix

    origin = frame.realize_frame(coord.UnitSphericalRepresentation(0*u.deg,
                                                                   0*u.deg))
    gc_icrs = origin.transform_to(coord.ICRS)

    if R is None:  # no simple matrix transformation:
        usph_band = coord.UnitSphericalRepresentation(
            np.random.uniform(0, 360, 10000)*u.deg,
            np.random.uniform(-1, 1, size=10000)*u.deg)
        band_icrs = frame.realize_frame(usph_band).transform_to(coord.ICRS)

        R1 = coord.matrix_utilities.rotation_matrix(gc_icrs.ra, 'z')
        R2 = coord.matrix_utilities.rotation_matrix(-gc_icrs.dec, 'y')
        Rtmp = R2 @ R1

        psi = get_roll(band_icrs, Rtmp, x0).degree

    else:
        origin = frame.realize_frame(coord.UnitSphericalRepresentation(0*u.deg,
                                                                       0*u.deg))
        gc_icrs = origin.transform_to(coord.ICRS)
        psi = -np.degrees(np.arctan2(R[2, 1], R[2, 2]))

    return [gc_icrs.ra.degree, gc_icrs.dec.degree, psi]


# Rotation vectors (Euler angles) for standard coordinate systems:
rots = dict()
rots['gal'] = [266.40498829, -28.93617776, 59.]
rots['gal-anticenter'] = [86.40498829, 28.93617776, -59.]
rots['sag'] = [284.03876751, -29.00408353, -10.]
rots['icrs'] = [180, 0]
