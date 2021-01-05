from .basemodel import BaseModel
import numpy as np
import numba as nb


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


@nb.njit(fastmath=True, cache=True)
def phiarray(x, y, modelargs, out, numexpr=False):
    b, x0, y0, e, te, s = modelargs
    c = np.cos(te * np.pi / 180)
    c2 = c * c
    s = np.sin(te * np.pi / 180)
    s2 = s * s
    sc = s * c

    # transformation from actual coords to natural coords
    # (the frame/axes are rotated so there is no ellipticity angle in the calculation).
    # This makes the expressions in the model modules simpler to calculate.
    xp = -s * (x - x0) + c * (y - y0)
    yp = -c * (x - x0) - s * (y - y0)

    if e < 1e-6:
        pot, px, py, pxx, pyy, pxy = spherical(xp, yp, modelargs)
    else:
        pot, px, py, pxx, pyy, pxy = elliptical(xp, yp, modelargs)

    out[0] += pot

    # Inverse transformation back into desired coordinates.
    out[1] += -s * px - c * py
    out[2] += c * px - s * py
    out[3] += s2 * pxx + c2 * pyy + 2 * sc * pxy
    out[4] += c2 * pxx + s2 * pyy - 2 * sc * pxy
    out[5] += sc * (pyy - pxx) + (s2 - c2) * pxy

    #return np.vstack((pot, new_phix, new_phiy, new_phixx, new_phiyy, new_phixy))

def SIE(*modelargs):
    return dotdict({'phiarray': phiarray, 'modelargs': modelargs})

@nb.njit(fastmath=True)
def elliptical(x, y, modelargs):
    b, x0, y0, e, te, s = modelargs[:6]

    x2 = x * x
    y2 = y * y
    s2 = s * s
    q = 1.0 - e
    q2 = q * q
    om = 1.0 - q2
    rt = np.sqrt(om)
    psi = np.sqrt(q2 * (s2 + x2) + y2)
    psis = psi + s

    phix = b * q / rt * np.arctan(rt * x / psis)
    phiy = b * q / rt * np.arctanh(rt * y / (psi + s * q2))

    invDenom = b * q / (psi * (om * x2 + psis * psis))
    phixx = (psi * psis - q2 * x2) * invDenom
    phiyy = (x2 + s * psis) * invDenom
    phixy = -x * y * invDenom

    pot = b * q * s * (-0.5 * np.log(psis * psis + om * x2) + np.log(s * (1.0 + q))) + x * phix + y * phiy

    return np.vstack((pot, phix, phiy, phixx, phiyy, phixy))


@nb.njit(fastmath=True)
def spherical(x, y, modelargs):
    b, x0, y0, e, te, s = modelargs[:6]

    rad = np.sqrt(x * x + y * y + s * s)
    sprad = s + rad
    invDenom = b / (rad * sprad * sprad)

    pot = b * (rad - s * (1 + np.log(sprad / (2 * s))))
    phix = b * x / sprad
    phiy = b * y / sprad
    phixx = (s * sprad + y * y) * invDenom
    phiyy = (s * sprad + x * x) * invDenom
    phixy = -x * y * invDenom

    return np.vstack((pot, phix, phiy, phixx, phiyy, phixy))
