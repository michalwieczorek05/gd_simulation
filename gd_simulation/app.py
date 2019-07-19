import numpy as np
from energy import Energy
from global_variables import MPl
from integrator import Integrator
from system_state import System_State

def InitialPrint(partial_state):
    print('The initial potential energy is:')
    print(np.average(Energy.potential_energy(partial_state))/pow(MPl, 4))
    print('Initial Phi Gradient energy is:')
    print(np.average(Energy.phi_gradient_energy(partial_state))/pow(MPl, 4))
    print('Initial Chi Gradient energy is:')
    print(np.average(Energy.chi_gradient_energy(partial_state))/pow(MPl, 4))
    print('Initial Phi Kinetic energy is:')
    print(np.average(Energy.phi_kinetic_energy(partial_state))/pow(MPl, 4))
    print('Initial Chi Kinetic energy is:')
    print(np.average(Energy.chi_kinetic_energy(partial_state))/pow(MPl, 4))


def FinalPrint(partial_state):
    print('The potential energy is:')
    print(np.average(Energy.potential_energy(partial_state))/pow(MPl, 4))
    print('Phi Gradient energy is:')
    print(np.average(Energy.phi_gradient_energy(partial_state))/pow(MPl, 4))
    print('Chi Gradient energy is:')
    print(np.average(Energy.chi_gradient_energy(partial_state))/pow(MPl, 4))
    print('Phi Kinetic energy is:')
    print(np.average(Energy.phi_kinetic_energy(partial_state))/pow(MPl, 4))
    print('Chi Kinetic energy is:')
    print(np.average(Energy.chi_kinetic_energy(partial_state))/pow(MPl, 4))

def main():
    system_state = System_State()
    system_state.initialize()
    InitialPrint(system_state.current_state)
    integrator = Integrator()
    integrator.write_run_informations()
    integrator.integrate(system_state)
    FinalPrint(system_state.current_state)

if __name__ == "__main__":
    main()





