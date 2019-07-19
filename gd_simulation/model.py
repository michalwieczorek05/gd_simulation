import numpy as np
from global_variables import MPl, specific_model_parameters
from utils import pw2

M = specific_model_parameters['M']
MInt = specific_model_parameters['MInt']
Mh = specific_model_parameters['Mh']
M *= MPl
MInt *= MPl
Mh *= MPl

class Model:

    @staticmethod
    def inflaton_potential(phi):
        return pow(M, 4.) * pw2(1. - np.exp(-np.sqrt(2. / 3.) * np.abs(phi) / MPl))

    @staticmethod
    def chi_potential(chi):
        return 0.5 * pw2(Mh) * pw2(chi)

    @staticmethod
    def potential(phi, chi):
        return Model.inflaton_potential(phi) + Model.chi_potential(chi)

    @staticmethod
    def d_phi_potential(phi, chi):
        return np.sign(phi) * pow(M, 4.) * (2. / MPl) * np.sqrt(2. / 3.) * np.exp(
            -np.sqrt(2. / 3.) * np.abs(phi) / MPl) * (1. - np.exp(-np.sqrt(2. / 3.) * np.abs(phi) / MPl))

    @staticmethod
    def d_chi_potential(phi, chi):
        return pw2(Mh) * chi

    @staticmethod
    def dd_phiphi_potential(phi, chi):
        return -pow(M, 4.) * (4. / (3. * pw2(MPl))) * (np.exp(-np.sqrt(2. / 3.) * np.abs(phi) / MPl) - 2. * np.exp(
            -2. * np.sqrt(2. / 3.) * np.abs(phi) / MPl))

    @staticmethod
    def dd_chichi_potential(phi, chi):
        return pw2(Mh)

    @staticmethod
    def non_canonical_multiplier(chi):
        return 1. + (2. / (MInt * MInt)) * pw2(chi)

    @staticmethod
    def d_non_canonical_multiplier(chi):
        return 4. * chi / (MInt * MInt)

    @staticmethod
    def dd_non_canonical_multiplier(chi):
        return 4. / (MInt * MInt)
