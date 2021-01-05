from .basemodel import BaseModel
import numpy as np
import numexpr as ne


if ne.use_vml:
    ne.set_vml_accuracy_mode('fast')

class Shear(object):
    def __init__(self, gamma1, gamma2, x0, y0):
        self.gamma1 = float(gamma1)
        self.gamma2 = float(gamma2)
        self.x0 = float(x0)
        self.y0 = float(y0)

    def modelargs(self):
        return [self.gamma1, self.gamma2, self.x0, self.y0]

    def phiarray(self, x, y, numexpr=True, *args, **kwargs):
        gamma1, gamma2, x0, y0 = self.modelargs()
        if numexpr:
            x_ = x - x0
            y_ = y - y0
            phi = ne.evaluate("0.5*(gamma1*x_*x_+2*gamma2*x_*y_-gamma1*y_*y_)")
            phix = ne.evaluate("gamma1*x_+gamma2*y_")
            phiy = ne.evaluate("gamma2*x_-gamma1*y_")
            kappa = 0.
            phixx = ne.evaluate("kappa+gamma1+0*x_")
            phiyy = ne.evaluate("kappa-gamma1+0*x_")
            phixy = ne.evaluate("gamma2+0*x_")
        else:
            x_ = x - x0
            y_ = y - y0
            phi = 0.5*(gamma1*x_*x_+2.*gamma2*x_*y_-gamma1*y_*y_)
            phix = gamma1*x_+gamma2*y_
            phiy = gamma2*x_-gamma1*y_
            kappa = 0.
            phixx = kappa + gamma1 + 0*x_
            phiyy = kappa - gamma1 + 0*x_
            phixy = gamma2 + 0*x_
        res = np.array((phi, phix, phiy, phixx, phiyy, phixy))
        return res
