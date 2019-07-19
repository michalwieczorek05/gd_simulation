import sys

import yaml


def load_config(config_path):
    with open(config_path, 'r') as stream:
        run_parameters = yaml.safe_load(stream)
    return run_parameters

run_parameters = load_config(sys.argv[1])
# Cut Off dependent parameters
CutOff = run_parameters['cutoff_dependent_parameters']['cutoff']
MM = run_parameters['cutoff_dependent_parameters']['rescaling_parameter']

# Model parameters
MPl = MM
phi0 = run_parameters['model_parameters']['phi0'] * MPl
d_phi0 = run_parameters['model_parameters']['d_phi0'] * MPl * MPl
chi0 = run_parameters['model_parameters']['chi0'] * MPl
d_chi0 = run_parameters['model_parameters']['d_chi0'] * MPl * MPl
a0 = run_parameters['model_parameters']['a0']
specific_model_parameters = run_parameters['model_parameters']['specific_model_parameters']


# Error analysis parameters
tau1 = run_parameters['error_analysis_parameters']['tau1']
tau2 = run_parameters['error_analysis_parameters']['tau2']
kappa1 = 0.7
kappa2 = 1.6

# run parameters
initial_time_step = run_parameters['technical_parameters']['initial_time_step']
N = run_parameters['technical_parameters']['N']
steps_number = run_parameters['technical_parameters']['steps_number']
continuation = run_parameters['technical_parameters']['continuation']
startIter = f"_iter={run_parameters['technical_parameters']['startIter']}"
write_separate_potential_energy = run_parameters['technical_parameters']['write_separate_potential_energy']


# names of different files
output_dir_path = run_parameters['technical_parameters']['output_dir_path']
SpecialComment = ''
Comment = f"_MInt={str(specific_model_parameters['MInt']).replace('.', ',')}" \
    f"_Mh={str(specific_model_parameters['Mh']).replace('.', ',')}_CO={CutOff}_N=" \
    f"{N}{SpecialComment}"
