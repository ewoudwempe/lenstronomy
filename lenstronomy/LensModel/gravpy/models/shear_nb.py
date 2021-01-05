from .basemodel import BaseModel
import numpy as np
import numba as nb

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


@nb.njit(fastmath=True, cache=True)
def phiarray(x, y, modelargs, out, numexpr=False):
    gamma1, gamma2, x0, y0 = modelargs
    x_ = x - x0
    y_ = y - y0
    out[0] += 0.5 * (gamma1 * x_ * x_ + 2. * gamma2 * x_ * y_ - gamma1 * y_ * y_)
    out[1] += gamma1 * x_ + gamma2 * y_
    out[2] += gamma2 * x_ - gamma1 * y_
    kappa = 0.
    out[3] += kappa + gamma1 + 0 * x_
    out[4] += kappa - gamma1 + 0 * x_
    out[5] += gamma2 + 0 * x_

    #return np.vstack((phi, phix, phiy, phixx, phiyy, phixy))

def Shear(*modelargs):
    return dotdict({'phiarray': phiarray, 'modelargs': modelargs})

