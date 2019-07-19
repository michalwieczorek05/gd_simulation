import numpy as np
from model import Model
from utils import pw2


class Energy:

    @staticmethod
    def grad2(field):
        return (pw2(np.roll(field, 1, axis=0) - field) + pw2(np.roll(field, -1, axis=0) - field) + pw2(
            np.roll(field, 1, axis=1) - field) + pw2(np.roll(field, -1, axis=1) - field) + pw2(
            np.roll(field, 1, axis=2) - field) + pw2(np.roll(field, -1, axis=2) - field)) / 2.

    @staticmethod
    def phi_gradient_term(partial_state):
        return Model.non_canonical_multiplier(partial_state.chi) * Energy.grad2(partial_state.phi) / 2.

    @staticmethod
    def chi_gradient_term(partial_state):
        return Energy.grad2(partial_state.chi) / 2.

    @staticmethod
    def phi_kinetic_term(partial_state):
        return partial_state.phiPi * partial_state.phiPi / Model.non_canonical_multiplier(partial_state.chi) / 2.

    @staticmethod
    def chi_kinetic_term(partial_state):
        return partial_state.chiPi * partial_state.chiPi / 2.

    @staticmethod
    def phi_gradient_energy(partial_state):
        return pow(partial_state.a, -2.) * Energy.phi_gradient_term(partial_state)

    @staticmethod
    def chi_gradient_energy(partial_state):
        return pow(partial_state.a, -2.) * Energy.chi_gradient_term(partial_state)

    @staticmethod
    def phi_kinetic_energy(partial_state):
        return pow(partial_state.a, -6.) * Energy.phi_kinetic_term(partial_state)

    @staticmethod
    def chi_kinetic_energy(partial_state):
        return pow(partial_state.a, -6.) * Energy.chi_kinetic_term(partial_state)

    @staticmethod
    def potential_energy(partial_state):
        return Model.potential(partial_state.phi, partial_state.chi)

    @staticmethod
    def inflaton_potential_energy(partial_state):
        return Model.inflaton_potential(partial_state.phi)

    @staticmethod
    def chi_potential_energy(partial_state):
        return Model.chi_potential(partial_state.chi)

    @staticmethod
    def full_energy(partial_state):
        energy = Energy.potential_energy(partial_state)
        energy += Energy.phi_gradient_energy(partial_state)
        energy += Energy.phi_kinetic_energy(partial_state)
        energy += Energy.chi_gradient_energy(partial_state)
        energy += Energy.chi_kinetic_energy(partial_state)
        return energy
