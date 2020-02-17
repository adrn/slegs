import os

from astropy.stats import sigma_clip
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np


def get_vscale(img, q=[5, 95]):
    _tmp = img.ravel()
    stdfunc = lambda x, axis: 1.5*np.median(np.abs(x-np.median(x, axis=axis)),
                                            axis=axis)
    _tmp_clipped = sigma_clip(_tmp[_tmp != 0], stdfunc=stdfunc)

    return np.percentile(_tmp_clipped[_tmp_clipped > 0], q)


def make_anim_frames(cube, distmods, proj, name,
                     mu_lim=None, vscale=None, sigma=None, figsize=None,
                     save_path=None, close_fig=True):
    """
    Parameters
    ----------
    cube : numpy.ndarray (n_pix, n_distmods)
        The full cube of filtered healpix maps.
    distmods : numpy.ndarray (n_distmods, )
        An array of distance modulus values for each layer in the cube.
    proj : healpy projection
        The healpy projection instance.
    name : str
        Stream name
    mu_lim : tuple (optional)
        The distance modulus limits to animate over. If not specified, this
        defaults to the full range in the cube.
    vscale : dict, tuple (optional)
        If a dictionary, this will be passed to get_vscale() as keyword
        arguments. If a tuple, this is assumed to be a tuple of (vmin, vmax)
        values to hard-set for all frames.
    sigma : float [radians] (optional)
        Smoothing.
    figsize : tuple (optional)
        If not set, this is determined automatically with the xsize and ysize of
        the projection.
    save_path : str, bool (optional)
        The path to save plots to. If set to False, this won't save plots.
    close_fig : bool (optional)
        Close the figure after saving.

    """

    if sigma is None:
        sigma = np.radians(0.12)  # MAGIC NUMBER

    if mu_lim is None:
        mu_lim = (distmods.min(), distmods.max())

    mu_idx = (np.argmin(np.abs(distmods - mu_lim[0])),
              np.argmin(np.abs(distmods - mu_lim[1])))

    if figsize is None:
        proj_info = proj.get_proj_plane_info()
        xsize = proj_info.get('xsize', 1.)
        ysize = proj_info.get('ysize', xsize)
        figsize = (10, 10 * ysize / xsize)

    if save_path is None:
        save_path = '.'

    # Hacks...
    if isinstance(vscale, dict) or vscale is None:
        vscale_func = get_vscale

        if vscale is None:
            vscale_kw = {}
        else:
            vscale_kw = vscale

    else:
        vscale_func = lambda *args, **_: vscale
        vscale_kw = {}

    nside = hp.npix2nside(cube.shape[0])
    func = lambda x, y, z: hp.vec2pix(nside, x, y, z)

    for j, i in enumerate(np.arange(mu_idx[0], mu_idx[1]+1, 1)):
        print(f'm-M = {distmods[i]:.1f}')

        hpxmap = cube[:, i]
        hpxmap_smooth = hp.smoothing(hpxmap, sigma=sigma, verbose=False)

        img = proj.projmap(hpxmap_smooth, func)

        vmin, vmax = vscale_func(img, **vscale_kw)
        print(f'vmin, vmax = ({vmin:.2f}, {vmax:.2f})')

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.imshow(img, origin='bottom', vmin=vmin, vmax=vmax, cmap='Greys',
                  extent=proj.get_extent())
        ax.set_title(f'm-M = {distmods[i]:.1f}')

        if close_fig is not False:
            fig.savefig(os.path.join(save_path, f'{name}_{j:02d}.png'), dpi=250)

        if close_fig:
            plt.close(fig)
