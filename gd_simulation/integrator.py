import math
import numpy as np
from energy import Energy
from global_variables import continuation, output_dir_path, Comment, N, MPl, tau1, tau2, kappa1, kappa2, CutOff, \
    initial_time_step, steps_number, write_separate_potential_energy, specific_model_parameters
from model import Model
from update_functions import dgradPhiTerm, paFirstUpdate, phiFirstUpdate, chiPiFirstUpdate, dgradChiTerm, \
    chiPiPartSecondUpdate
from utils import pw2


class Integrator:

    def __init__(self):
        self.files_with_data = self.create_files_dict()

    def create_files_dict(self):
        files_dict = {}
        if continuation:
            files_dict['EfoldsBoxPlotList'] = open(f"{output_dir_path}EfoldsBoxPlotList{Comment}.dat", 'a')
            files_dict['dtList'] = open(f"{output_dir_path}dtList{Comment}.dat", 'a')
            files_dict['energyData'] = open(f"{output_dir_path}energyData{Comment}.dat", 'a')
            files_dict['fieldData'] = open(f"{output_dir_path}fieldData{Comment}.dat", 'a')
            files_dict['chiMagnitudeData'] = open(f"{output_dir_path}chiMagnitudeData{Comment}.dat", 'a')
        else:
            files_dict['EfoldsBoxPlotList'] = open(f"{output_dir_path}EfoldsBoxPlotList{Comment}.dat", 'w')
            files_dict['dtList'] = open(f"{output_dir_path}dtList{Comment}.dat", 'w')
            files_dict['energyData'] = open(f"{output_dir_path}energyData{Comment}.dat", 'w')
            files_dict['fieldData'] = open(f"{output_dir_path}fieldData{Comment}.dat", 'w')
            files_dict['chiMagnitudeData'] = open(f"{output_dir_path}chiMagnitudeData{Comment}.dat", 'w')
        return files_dict

    def make_2nd_order_time_step(self, partial_state, h):
        partial_state.a += (h / 2.) * (-partial_state.pa / (6. * pow(N, 3) * pow(MPl, 2)))

        partial_state.pa += paFirstUpdate(h, partial_state.phiPi, partial_state.chi, partial_state.a)
        partial_state.phi += phiFirstUpdate(h, partial_state.a, partial_state.phiPi, partial_state.chi)
        partial_state.chiPi += chiPiFirstUpdate(h, partial_state.a, partial_state.phiPi, partial_state.chi)

        partial_state.pa += (h / 2.) * np.sum(pw2(partial_state.chiPi)) / pow(partial_state.a, 3)
        partial_state.chi += (h / (2. * pw2(partial_state.a))) * partial_state.chiPi

        partial_state.pa -= (h) * pow(N * partial_state.a, 3.) * (2. * (np.average(Energy.phi_gradient_energy(partial_state)) +
                                                                        np.average(Energy.chi_gradient_energy(
                                                                            partial_state))) + 4. * np.average(Model.potential(
            partial_state.phi, partial_state.chi)))
        partial_state.phiPi -= (h) * (
                    pow(partial_state.a, 4) * Model.d_phi_potential(partial_state.phi, partial_state.chi) +
                    pow(partial_state.a, 2) * dgradPhiTerm(partial_state.phi, partial_state.chi) / 2.)
        partial_state.chiPi -= (h) * (
                    pow(partial_state.a, 4) * Model.d_chi_potential(partial_state.phi, partial_state.chi) +
                    pow(partial_state.a, 2) * (dgradChiTerm(partial_state.chi)
                                               / 2. + chiPiPartSecondUpdate(partial_state.chi) * Energy.grad2(
                        partial_state.phi)))

        partial_state.chi += (h / (2. * pw2(partial_state.a))) * partial_state.chiPi
        partial_state.pa += (h / 2.) * np.sum(pw2(partial_state.chiPi)) / pow(partial_state.a, 3)

        partial_state.chiPi += chiPiFirstUpdate(h, partial_state.a, partial_state.phiPi, partial_state.chi)
        partial_state.phi += phiFirstUpdate(h, partial_state.a, partial_state.phiPi, partial_state.chi)
        partial_state.pa += paFirstUpdate(h, partial_state.phiPi, partial_state.chi, partial_state.a)

        partial_state.a += (h / 2.) * (-partial_state.pa / (6. * pow(N, 3) * pow(MPl, 2)))

    def make_4th_order_time_step(self, partial_state, h):
        z = (2. ** (1 / 3.))
        z0 = -z / (2. - z)
        z1 = 1. / (2. - z)
        self.make_2nd_order_time_step(partial_state, z1 * h)
        self.make_2nd_order_time_step(partial_state, z0 * h)
        self.make_2nd_order_time_step(partial_state, z1 * h)

    def error_estimation(self, current_state, error_computing_state):
        error_a = np.abs((current_state.a - error_computing_state.a) / current_state.a)
        error_pa = np.abs((current_state.pa - error_computing_state.pa) / current_state.pa)
        error_phi = np.average(np.abs((current_state.phi - error_computing_state.phi) / current_state.phi))
        error_chi = np.average(np.abs((current_state.chi - error_computing_state.chi) / current_state.chi))
        error_phiPi = np.average(np.abs((current_state.phiPi - error_computing_state.phiPi) / current_state.phiPi))
        error_chiPi = np.average(np.abs((current_state.chiPi - error_computing_state.chiPi) / current_state.chiPi))
        return error_a + error_pa + error_phi + error_chi + error_phiPi + error_chiPi

    def write_curent_data(self, partial_state, step_num, h):
        self.files_with_data['energyData'].write(repr(partial_state.a))
        self.files_with_data['energyData'].write('	')
        self.files_with_data['energyData'].write(
            repr(np.average(Energy.phi_kinetic_energy(partial_state)) / pow(MPl, 4)))
        self.files_with_data['energyData'].write('	')
        self.files_with_data['energyData'].write(
            repr(np.average(Energy.chi_kinetic_energy(partial_state)) / pow(MPl, 4)))
        self.files_with_data['energyData'].write('	')
        self.files_with_data['energyData'].write(
            repr(np.average(Energy.phi_gradient_energy(partial_state)) / pow(MPl, 4)))
        self.files_with_data['energyData'].write('	')
        self.files_with_data['energyData'].write(
            repr(np.average(Energy.chi_gradient_energy(partial_state)) / pow(MPl, 4)))
        self.files_with_data['energyData'].write('	')
        if write_separate_potential_energy:
            self.files_with_data['energyData'].write(
                repr(np.average(Energy.inflaton_potential_energy(partial_state)) / pow(MPl, 4)))
            self.files_with_data['energyData'].write('	')
            self.files_with_data['energyData'].write(
                repr(np.average(Energy.chi_potential_energy(partial_state)) / pow(MPl, 4)))
            self.files_with_data['energyData'].write('\n')
        else:
            self.files_with_data['energyData'].write(
                repr(np.average(Energy.potential_energy(partial_state)) / pow(MPl, 4)))
            self.files_with_data['energyData'].write('\n')

        # writing fields in program units:
        self.files_with_data['fieldData'].write(repr(partial_state.a))
        self.files_with_data['fieldData'].write('	')
        self.files_with_data['fieldData'].write(repr(partial_state.pa))
        self.files_with_data['fieldData'].write('	')
        self.files_with_data['fieldData'].write(repr(np.average(partial_state.phi)))
        self.files_with_data['fieldData'].write('	')
        self.files_with_data['fieldData'].write(repr(np.average(partial_state.chi)))
        self.files_with_data['fieldData'].write('	')
        self.files_with_data['fieldData'].write(repr(np.average(partial_state.phiPi)))
        self.files_with_data['fieldData'].write('	')
        self.files_with_data['fieldData'].write(repr(np.average(partial_state.chiPi)))
        self.files_with_data['fieldData'].write('\n')
        self.files_with_data['chiMagnitudeData'].write(repr(partial_state.a))
        self.files_with_data['chiMagnitudeData'].write('	')
        self.files_with_data['chiMagnitudeData'].write(repr(np.sqrt(np.average(pw2(partial_state.chi)))))
        self.files_with_data['chiMagnitudeData'].write('\n')

        self.files_with_data['dtList'].write(repr(step_num))
        self.files_with_data['dtList'].write(' ')
        self.files_with_data['dtList'].write(repr(h))
        self.files_with_data['dtList'].write('\n')

    def write_final_data(self, partial_state, step_num):
        np.save(f"{output_dir_path}finalPhiFieldData{Comment}_iter={step_num}", partial_state.phi)
        np.save(f"{output_dir_path}finalChiFieldData{Comment}_iter={step_num}", partial_state.chi)
        np.save(f"{output_dir_path}finalPhiPiFieldData{Comment}_iter={step_num}", partial_state.phiPi)
        np.save(f"{output_dir_path}finalChiPiFieldData{Comment}_iter={step_num}", partial_state.chiPi)
        np.save(f"{output_dir_path}finalaPiaData{Comment}_iter={step_num}",
                np.array([partial_state.a, partial_state.pa]))

    def write_full_energy_data(self, partial_state):
        efolds = str('%.5f' % math.log(partial_state.a)).replace('.', ',')
        np.save(f"{output_dir_path}PeriodicEnergyData_Efolds={efolds}{Comment}",
                Energy.full_energy(partial_state))

    def write_full_field_data(self, partial_state):
        efolds = str('%.5f' % math.log(partial_state.a)).replace('.', ',')
        np.save(f"{output_dir_path}PeriodicPhiFieldData_Efolds={efolds}{Comment}", partial_state.phi)
        np.save(f"{output_dir_path}PeriodicChiFieldData_Efolds={efolds}{Comment}", partial_state.chi)

    def write_efolds_for_periodic_full_data(self, partial_state):
        efolds = str('%.5f' % math.log(partial_state.a)).replace('.', ',')
        self.files_with_data['EfoldsBoxPlotList'].write(efolds)
        self.files_with_data['EfoldsBoxPlotList'].write('	')

    def write_run_informations(self):
        InfData = open(f"{output_dir_path}RunInformation{Comment}.dat", 'w')
        InfData.write('N=' + repr(N))
        InfData.write('\n')
        InfData.write('CutOff=' + repr(CutOff))
        InfData.write('\n')
        InfData.write('tau1=' + repr(tau1))
        InfData.write('\n')
        InfData.write('dt=' + repr(initial_time_step))
        InfData.write('\n')
        InfData.write('specific_model_parameters' + repr(specific_model_parameters))
        InfData.write('\n')

    def integrate(self, system_state):
        h = initial_time_step
        for step_num in range(steps_number):
            if (step_num % 100) == 0:
                print(step_num)
            if (step_num % 1000) == 0:
                self.write_full_energy_data(system_state.current_state)
                self.write_full_field_data(system_state.current_state)
                self.write_efolds_for_periodic_full_data(system_state.current_state)
            if (step_num % 10000 == 0) and step_num > 0:
                self.write_final_data(system_state.current_state, step_num)

            system_state.make_error_state_assignment()
            system_state.make_new_to_old_assignment()
            self.make_4th_order_time_step(system_state.current_state, h)
            self.make_2nd_order_time_step(system_state.error_computing_state, h)
            error = self.error_estimation(system_state.current_state, system_state.error_computing_state)
            if error > tau1:
                h = h * ((tau1 / error) ** (1. / 3.)) * kappa1
                system_state.make_old_to_new_assignment()
            elif error < tau2:
                h = h * ((tau2 / error) ** (1. / 3.)) * kappa2
            self.write_curent_data(system_state.current_state, step_num, h)
        self.write_final_data(system_state.current_state, steps_number)

