__author__ = 'aymgal'

import numpy as np

from lenstronomy.Sampling.Samplers.base_nested_sampler import NestedSampler
import lenstronomy.Util.sampling_util as utils
import dynesty
from dynesty import utils as dyfunc

__all__ = ['DynestySampler']


class DynestySampler(NestedSampler):
    """
    Wrapper for dynamical nested sampling algorithm Dynesty by J. Speagle
    
    paper : https://arxiv.org/abs/1904.02180
    doc : https://dynesty.readthedocs.io/
    """

    def __init__(self, likelihood_module, prior_type='uniform', 
                 prior_means=None, prior_sigmas=None, width_scale=1, sigma_scale=1,
                 bound='multi', sample='auto', use_mpi=False, static=False,
                 pool=None, kwargs={}):
        """
        :param likelihood_module: likelihood_module like in likelihood.py (should be callable)
        :param prior_type: 'uniform' of 'gaussian', for converting the unit hypercube to param cube
        :param prior_means: if prior_type is 'gaussian', mean for each param
        :param prior_sigmas: if prior_type is 'gaussian', std dev for each param
        :param width_scale: scale the widths of the parameters space by this factor
        :param sigma_scale: if prior_type is 'gaussian', scale the gaussian sigma by this factor
        :param bound: specific to Dynesty, see https://dynesty.readthedocs.io
        :param sample: specific to Dynesty, see https://dynesty.readthedocs.io
        :param use_mpi: Use MPI computing if `True`
        :param kwargs: Any additional kwargs passed to the Sampler object, see https://dynesty.readthedocs.io
        """
        self._check_install()
        super(DynestySampler, self).__init__(likelihood_module, prior_type, 
                                             prior_means, prior_sigmas,
                                             width_scale, sigma_scale)

        if use_mpi and pool is None:
            from schwimmbad import MPIPool
            import sys
            pool = MPIPool(use_dill=True)  # use_dill=True not supported for some versions of schwimmbad
            if not pool.is_master():
                pool.wait()
                sys.exit(0)

        # create the Dynesty sampler
        sampler = dynesty.NestedSampler if static else dynesty.DynamicNestedSampler
        self._sampler = sampler(self.log_likelihood, self.prior, self.n_dims, pool=pool, **kwargs)
        self._has_warned = False


    def log_likelihood(self, x):
        """
        compute the log-likelihood given list of parameters

        :param x: parameter values
        :return: log-likelihood (from the likelihood module)
        """
        logL = self._ll(x)
        if not np.isfinite(logL):
            if not self._has_warned:
                print("WARNING : logL is not finite : return very low value instead")
            logL = -1e15
            self._has_warned = True
        return float(logL)

    def run(self, kwargs_run):
        """
        run the Dynesty nested sampler

        see https://dynesty.readthedocs.io for content of kwargs_run

        :param kwargs_run: kwargs directly passed to DynamicNestedSampler.run_nested
        :return: samples, means, logZ, logZ_err, logL, results
        """
        print("prior type :", self.prior_type)
        print("parameter names :", self.param_names)
    
        self._sampler.run_nested(**kwargs_run)

        results = self._sampler.results
        samples_w = results.samples  # weighted samples
        logL = results.logl
        logZ = results.logz
        logZ_err = results.logzerr

        # Compute weighted mean and covariance.
        weights = np.exp(results.logwt - logZ[-1])  # normalized weights
        if np.sum(weights) != 1.:
            # TODO : clearly this is not optimal...
            # weights should by definition be normalized, but it appears that for very small 
            # number of live points (typically in test routines), 
            # it is not *quite* the case (up to 6 decimals)
            weights = weights / np.sum(weights)

        means, covs = dyfunc.mean_and_cov(samples_w, weights)

        # Resample weighted samples to get equally weighted (aka unweighted) samples
        samples = dyfunc.resample_equal(samples_w, weights)

        return samples, means, logZ, logZ_err, logL, results

    def _check_install(self):

        try:
            import dynesty
            import dynesty.utils as dyfunc
        except:
            print("Warning : dynesty not properly installed (results might be unexpected). \
                    You can get it with $pip install dynesty.")
            self._dynesty_installed = False
        else:
            self._dynesty_installed = True
