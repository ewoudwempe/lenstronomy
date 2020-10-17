__author__ = 'aymgal'

import numpy as np

from lenstronomy.Sampling.Samplers.base_nested_sampler import NestedSampler
import lenstronomy.Util.sampling_util as utils
import ultranest

class UltranestSampler(NestedSampler):
    """
    Wrapper for dynamical nested sampling algorithm ultranest
    
    paper : TODO
    doc : https://ultranest.readthedocs.io/
    """

    def __init__(self, likelihood_module, prior_type='uniform', 
                 prior_means=None, prior_sigmas=None, width_scale=1, sigma_scale=1, static=False, kwargs={}):
        """
        :param likelihood_module: likelihood_module like in likelihood.py (should be callable)
        :param prior_type: 'uniform' of 'gaussian', for converting the unit hypercube to param cube
        :param prior_means: if prior_type is 'gaussian', mean for each param
        :param prior_sigmas: if prior_type is 'gaussian', std dev for each param
        :param width_scale: scale the widths of the parameters space by this factor
        :param sigma_scale: if prior_type is 'gaussian', scale the gaussian sigma by this factor
        :param kwargs: Any additional kwargs passed to the Sampler object, see https://ultranest.readthedocs.io
        """
        self._check_install()
        super(UltranestSampler, self).__init__(likelihood_module, prior_type, 
                                             prior_means, prior_sigmas,
                                             width_scale, sigma_scale)

        # create the ultranest sampler
        sampler = ultranest.NestedSampler if static else ultranest.ReactiveNestedSampler 

        self._sampler = sampler(self.param_names, self.log_likelihood, self.prior, **kwargs)
        self._has_warned = False

    def prior(self, u):
        """
        compute the mapping between the unit cube and parameter cube

        :param u: unit hypercube, sampled by the algorithm
        :return: hypercube in parameter space
        """
        if self.prior_type == 'gaussian':
            p = utils.cube2args_gaussian(u, self.lowers, self.uppers,
                                         self.means, self.sigmas, self.n_dims,
                                         copy=True)
        elif self.prior_type == 'uniform':
            p = utils.cube2args_uniform(u, self.lowers, self.uppers, 
                                        self.n_dims, copy=True)
        else:
            raise ValueError('prior type %s not supported! Chose "gaussian" or "uniform".')
        return p

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
        run the ultranest nested sampler

        see https://ultranest.readthedocs.io for content of kwargs_run

        :param kwargs_run: kwargs directly passed to NestedSampler.run
        :return: samples, means, logZ, logZ_err, logL, results
        """
        print("prior type :", self.prior_type)
        print("parameter names :", self.param_names)
    
        self._sampler.run(**kwargs_run)

        results = self._sampler.results
        samples_w = results['weighted_samples']['points']  # weighted samples
        logL = results['weighted_samples']['logl']
        logZ = results['logz']
        logZ_err = results['logzerr']

        # Compute weighted mean and covariance.
        weights = results['weighted_samples']['weights'] # normalized weights

        means = results['posterior']['mean']

        # Resample weighted samples to get equally weighted (aka unweighted) samples
        samples = ultranest.utils.resample_equal(samples_w, weights)

        return samples, means, logZ, logZ_err, logL, results

    def _check_install(self):

        try:
            import ultranest
        except:
            print("Warning : ultranest not properly installed (results might be unexpected). \
                    You can get it with $pip install ultranest.")
            self._ultranest_installed = False
        else:
            self._ultranest_installed = True
