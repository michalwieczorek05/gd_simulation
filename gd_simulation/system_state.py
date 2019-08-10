import copy

import math
import numpy as np
import os
from global_variables import continuation, output_dir_path, Comment, startIter, MPl, phi0, chi0, d_phi0, d_chi0, a0, N
from initialize import Initialize
from model import Model


class Partial_State:

    def __init__(self):
        self.a = 0.
        self.pa = 0.
        self.phi = []
        self.phiPi = []
        self.chi = []
        self.chiPi = []


class System_State:

    def __init__(self):
        self.current_state = Partial_State()
        self.error_computing_state = Partial_State()
        self.old_state = Partial_State()

    def _scale_factor_initialize(self):
        if continuation:
            self.current_state.a, self.current_state.pa = np.load(os.path.join(output_dir_path, f'finalaPiaData{Comment}{startIter}.npy'))
        else:
            self.current_state.a = a0
            self.current_state.pa = -6. * pow(N, 3) * MPl * math.sqrt((pow(d_phi0, 2) / 2. + Model.potential(phi0, chi0)) / 3.)

    def _phi_initialize(self):
        if continuation:
            self.current_state.phi = np.load(os.path.join(output_dir_path, f'finalPhiFieldData{Comment}{startIter}.npy'))
            self.current_state.phiPi = np.load(os.path.join(output_dir_path, f'finalPhiPiFieldData{Comment}{startIter}.npy'))
        else:
            m2Phi = Model.dd_phiphi_potential(phi0, chi0) - 2. * (pow(d_phi0, 2) / 2. + Model.potential(phi0, chi0)) / (3. * pow(MPl, 2.))
            self.current_state.phi, self.current_state.phiPi = Initialize.initialize_field_non_zero_modes(m2Phi)
            self.current_state.phi = np.array(self.current_state.phi)
            self.current_state.phiPi = np.array(self.current_state.phiPi)
            self.current_state.phi += phi0
            self.current_state.phiPi += d_phi0

    def _chi_initialize(self):
        if continuation:
            self.current_state.chi = np.load(
                os.path.join(output_dir_path, f'finalChiFieldData{Comment}{startIter}.npy'))
            self.current_state.chiPi = np.load(
                os.path.join(output_dir_path, f'finalChiPiFieldData{Comment}{startIter}.npy'))
        else:
            m2Chi = Model.dd_chichi_potential(phi0, chi0) - 2. * (pow(d_phi0, 2) / 2. + Model.potential(phi0, chi0)) / (3. * pow(MPl, 2.)) - Model.dd_non_canonical_multiplier(chi0)*pow(d_phi0, 2)/2.
            self.current_state.chi, self.current_state.chiPi = Initialize.initialize_field_non_zero_modes(m2Chi)
            self.current_state.chi = np.array(self.current_state.chi)
            self.current_state.chiPi = np.array(self.current_state.chiPi)
            self.current_state.chi += chi0
            self.current_state.chiPi = Initialize.mod_chi_pi_ini(self.current_state.chiPi, self.current_state.chi)
            self.current_state.chiPi += d_chi0

    def initialize(self):
        self._scale_factor_initialize()
        self._phi_initialize()
        self._chi_initialize()

    def make_error_state_assignment(self):
        self.error_computing_state = copy.deepcopy(self.current_state)

    def make_old_to_new_assignment(self):
        self.current_state = copy.deepcopy(self.old_state)

    def make_new_to_old_assignment(self):
        self.old_state = copy.deepcopy(self.current_state)

