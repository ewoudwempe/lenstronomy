import numpy as np
from lenstronomy.LensModel.gravpy import Gravlens
from lenstronomy.LensModel.gravpy import models
from lenstronomy.Util.param_util import ellipticity2phi_q

cart2pol = lambda x: (np.sqrt(x[0]**2+x[1]**2), np.arctan2(x[1],x[0])%(2*np.pi))
pol2cart = lambda x: (x[0]*np.cos(x[1]), x[0]*np.sin(x[1]))


class GravlensOverloaded(Gravlens):
    def __init__(self, lensModel, kwargs_lens, carargs, polarargs, make_dpoints=True, overload_lenscalcs=True, **kwargs_gravlens):
        self.lensModel = lensModel
        self.kwargs_lens = kwargs_lens
        self.transformed = None # For checking if we generated caustics later
        self.caustics = None

        super().__init__(carargs, polarargs, None, image=None, show_plot=False, **kwargs_gravlens)
        self.make_dpoints = make_dpoints
        self.overload_lenscalcs = overload_lenscalcs
        if overload_lenscalcs:
            self.relation = self.relation_ls
            self.magnification = self.magnification_ls
            self.mapping = self.mapping_ls
            self.carmapping = self.carmapping_ls
        else:
            if set(lensModel.lens_model_list) > {'SIE', 'SHEAR'}:
                raise ValueError("Only SIE, Shear supported for the gravlens builtin calculations")
            self.modelargs = []
            for i, mod in enumerate(lensModel.lens_model_list):
                if mod == 'SIE':
                    phi, q = ellipticity2phi_q(kwargs_lens[i]['e1'], kwargs_lens[i]['e2'])
                    self.modelargs.append(models.SIE(kwargs_lens[i]['theta_E']/np.sqrt(q), kwargs_lens[i]['center_x'], kwargs_lens[i]['center_y'],1-q, (np.pi/2+phi)*180/np.pi, 1e-6))
                if mod == 'SHEAR':
                    self.modelargs.append(models.Shear(kwargs_lens[i]['gamma1'], kwargs_lens[i]['gamma2'], kwargs_lens[i]['ra_0'], kwargs_lens[i]['dec_0']))


    def relation_ls(self, x, y):
        mags = self.lensModel.magnification(x, y, self.kwargs_lens)
        return np.sign(mags)

    def magnification_ls(self, x, y):
        mags = self.lensModel.magnification(x, y, self.kwargs_lens)
        return mags

    def mapping_ls(self, v):
        x_guess, y_guess = v
        x_mapped, y_mapped = self.lensModel.ray_shooting(x_guess, y_guess, self.kwargs_lens) # Near the center, this code fails, where Keeton's works.
        f_xx, f_xy, f_yx, f_yy = self.lensModel.hessian(x_guess, y_guess, self.kwargs_lens)
        DistMatrix = np.array([[1 - f_yy, -f_yx], [-f_xy, 1 - f_xx]]) # - for convention of keeton
        x_source, y_source = self.image
        deflectionvector = np.array([x_mapped - x_source, y_mapped - y_source])
        mine = [deflectionvector, DistMatrix]
        return mine

    def minmapping(self, v):
        m, h = self.mapping(v)
        toret = m[0]**2+m[1]**2, 2 * h[::-1,::-1].dot(m)
        return toret

    def carmapping_ls(self, x, y):
        return np.array(self.lensModel.ray_shooting(x, y, self.kwargs_lens)).T

    def get_caustics(self):
        args = self.generate_ranges()
        x, y = args[0]
        polargrids = args[1]
        self.transformations((x, y), polargrids)

        a = np.moveaxis(self.critical_cells, -1, 0)
        mags = self.magnification(a[0].flatten(), a[1].flatten()).reshape(a.shape[1:])
        fil = np.abs(mags[:].min(axis=-1)) > 0.5
        means = self.critical_cells.mean(axis=1)[fil]
        if 'center_x' in self.kwargs_lens[0]:
            means -= means.mean(axis=0) # these are only used for proper azimuthal ordering around the center.
        else:
            print("warning: not able to properly center the image before calculating the critical curves.")

        rs, phis = cart2pol(np.moveaxis(means, -1, 0))
        idx_sort = np.argsort(phis)
        segments = marching_squares(self.critical_cells[fil][idx_sort], 1 / mags[fil][idx_sort])
        self.segments = segments
        crit_line = _assemble_contours(segments)
        crit_line = np.concatenate([c[:-1] if i!=len(crit_line)-1 else c for i, c in enumerate(crit_line)])
        x_s, y_s = self.lensModel.ray_shooting(*crit_line.T, self.kwargs_lens)
        return np.array([*crit_line.T, x_s, y_s])

    def solve(self, x_source, y_source, ret_caustics=False):
        self.image = np.array([x_source, y_source])
        if self.caustics is not None:
            self.find_source()
        else:
            self.run()
        if ret_caustics:
            return self.realpos, self.get_caustics()
        else:
            return self.realpos

    def get_guess_positions(self, x_source, y_source):
        self.image = np.array([x_source, y_source])

        if self.caustics is not None:
            guesses = self.find_source(onlyguess=True)
        else:
            guesses = self.run(onlyguess=True)
        return guesses

    def validate_arguments(self):
        pass
