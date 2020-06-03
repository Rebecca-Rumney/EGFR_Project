import numpy as np
import scipy.constants
from scipy.optimize import fsolve
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from SALib.sample import fast_sampler
from SALib.analyze import fast
from SALib.test_functions import Ishigami
from ODE_solver import ODESolving as ode


def evaluate_model(param_values, times, cycles: int):
    param_dict = dict(zip(parameters, param_values))
    if cycles == 3:
        solver = ode(params=param_dict, verbose=False).solve(verbose=False, times=times, equations='coupled3')
        return solver.y[6, :]
    elif cycles == 4:
        solver = ode(params=param_dict, verbose=False).solve(verbose=False, times=times, equations='coupled4')
        return solver.y[7, :]
    elif cycles == 2:
        solver = ode(params=param_dict, verbose=False).solve(verbose=False, times=times, equations='coupled2')
        return solver.y[5, :]


def cycle_3_sensitivity(parameters, exponential_level, parameter_base_values, cycles=None, verbose=False):

    if cycles is None:
        cycles = [3]
    bounds = [[-exponential_level, exponential_level]] * len(parameters)
    problem = {
        'num_vars': len(parameters),
        'names': parameters,
        'bounds': bounds
    }

    N = 1000
    param_exponents = fast_sampler.sample(problem, N)
    t = np.linspace(0, 180, 1801)
    Y = np.zeros([param_exponents.shape[0], t.shape[0]])

    if 3 in cycles:
        for i, X in enumerate(param_exponents):
            if verbose:
                print('Solving Equation', i + 1, '/', len(param_exponents))
            param_values = np.asarray(parameter_base_values) * np.power(10, X)
            Y[i] = evaluate_model(param_values, t, 3)

        sensitivities1_3cycles = [None] * len(t)
        sensitivitiesT_3cycles = [None] * len(t)
        for i in range(0, len(t)):
            if verbose:
                print('Analysing Sensitivities at time', t[i], '/', t[-1])
            Si = fast.analyze(problem, Y[:, i])
            sensitivities1_3cycles[i] = Si['S1']
            sensitivitiesT_3cycles[i] = Si['ST']

        for i in range(0, len(parameters)):
            plt.plot(t[:], np.asarray(sensitivities1_3cycles)[:, i], label=parameters[i])

        plt.xlabel('t')
        plt.ylabel('Sensitivity')
        plt.savefig('images/Sensitivity1_3enzymes.pdf')
        plt.show()

        print('First Sensitivity peaks for 3 cycles:')
        for i in range(0, len(parameters)):
            print('\t', parameters[i], 'peaks:')
            peak_list, _ = find_peaks(np.asarray(sensitivities1_3cycles)[100:, i])
            peak_list = np.sort(peak_list)
            for peak in peak_list:
                print('\t\t at t =', t[peak], 'for a sensitivity of', np.asarray(sensitivities1_3cycles)[peak, i])

        for i in range(0, len(parameters)):
            plt.plot(t[:], np.asarray(sensitivitiesT_3cycles)[:, i], label=parameters[i])

        plt.xlabel('t')
        plt.ylabel('Sensitivity')
        plt.savefig('images/SensitivityT_3enzymes.pdf')
        plt.show()


        print('Total Sensitivity peaks for 3 cycles:')
        for i in range(0, len(parameters)):
            print('\t', parameters[i], 'peaks:')
            peak_list, _ = find_peaks(np.asarray(sensitivitiesT_3cycles)[100:, i])
            peak_list = np.sort(peak_list)
            for peak in peak_list:
                print('\t\t at t =', t[peak], 'for a sensitivity of', np.asarray(sensitivitiesT_3cycles)[peak, i])


        for i in range(0, len(parameters)):
            plt.plot(t[:], np.asarray(sensitivities1_3cycles)[:, i] / max(np.asarray(sensitivities1_3cycles)[1:, i]), label=parameters[i])

        plt.xlabel('t')
        plt.ylabel('sensitivity')
        plt.yticks([])
        plt.savefig('images/SensitivityScaled1_3enzymes.pdf')
        plt.show()

    if 4 in cycles:
        for i, X in enumerate(param_exponents):
            if verbose:
                print('Solving Equation', i + 1, '/', len(param_exponents))
            param_values = np.asarray(parameter_base_values) * np.power(10, X)
            Y[i] = evaluate_model(param_values, t, 4)

        sensitivities1_4cycles = [None] * len(t)
        sensitivitiesT_4cycles = [None] * len(t)
        for i in range(0, len(t)):
            # print('Analysing Sensitivities at time', t[i], '/', t[-1])
            Si = fast.analyze(problem, Y[:, i])
            sensitivities1_4cycles[i] = Si['S1']
            sensitivitiesT_4cycles[i] = Si['ST']

        for i in range(0, len(parameters)):
            plt.plot(t[:], np.asarray(sensitivities1_4cycles)[:, i], label=parameters[i])

        plt.xlabel('t')
        plt.ylabel('Sensitivity')
        plt.savefig('images/Sensitivity1_4enzymes.pdf')
        plt.show()


        print('First Sensitivity peaks for 4 cycles:')
        for i in range(0, len(parameters)):
            print('\t', parameters[i], 'peaks:')
            peak_list, _ = find_peaks(np.asarray(sensitivities1_4cycles)[100:, i])
            peak_list = np.sort(peak_list)
            for peak in peak_list:
                print('\t\t at t =', t[peak], 'for a sensitivity of', np.asarray(sensitivities1_4cycles)[peak, i])

        for i in range(0, len(parameters)):
            plt.plot(t[:], np.asarray(sensitivitiesT_4cycles)[:, i], label=parameters[i])

        plt.xlabel('t')
        plt.ylabel('Sensitivity 4 cycle')
        plt.savefig('images/SensitivityT_4enzymes.pdf')
        plt.show()

        print('Total Sensitivity peaks for 4 cycles:')
        for i in range(0, len(parameters)):
            print('\t', parameters[i], 'peaks:')
            peak_list, _ = find_peaks(np.asarray(sensitivitiesT_4cycles)[100:, i])
            peak_list = np.sort(peak_list)
            for peak in peak_list:
                print('\t\t at t =', t[peak], 'for a sensitivity of', np.asarray(sensitivitiesT_4cycles)[peak, i])

        for i in range(0, len(parameters)):
            plt.plot(t[:], np.asarray(sensitivities1_4cycles)[:, i] / max(np.asarray(sensitivities1_4cycles)[1:, i]),
                     label=parameters[i])

        plt.xlabel('t')
        plt.ylabel('sensitivity')
        plt.yticks([])
        plt.savefig('images/SensitivityScaled1_4enzymes.pdf')
        plt.show()


    if 2 in cycles:
        t = np.linspace(0, 180, 1801)
        Y = np.zeros([param_exponents.shape[0], t.shape[0]])

        for i, X in enumerate(param_exponents):
            if verbose:
                print('Solving Equation', i + 1, '/', len(param_exponents))
            param_values = np.asarray(parameter_base_values) * np.power(10, X)
            Y[i] = evaluate_model(param_values, t, 2)

        sensitivities1_2cycles = [None] * len(t)
        sensitivitiesT_2cycles = [None] * len(t)
        for i in range(0, len(t)):
            # print('Analysing Sensitivities at time', t[i], '/', t[-1])
            Si = fast.analyze(problem, Y[:, i])
            sensitivities1_2cycles[i] = Si['S1']
            sensitivitiesT_2cycles[i] = Si['ST']

        for i in range(0, len(parameters)):
            plt.plot(t[:], np.asarray(sensitivities1_2cycles)[:, i], label=parameters[i])

        plt.xlabel('t')
        plt.ylabel('Sensitivity')
        plt.savefig('images/Sensitivity1_2enzymes.pdf')
        plt.show()

        print('First Sensitivity peaks for 2 cycles:')
        for i in range(0, len(parameters)):
            print('\t', parameters[i], 'peaks:')
            peak_list, _ = find_peaks(np.asarray(sensitivities1_2cycles)[100:, i])
            peak_list = np.sort(peak_list)
            for peak in peak_list:
                print('\t\t at t =', t[peak], 'for a sensitivity of', np.asarray(sensitivities1_2cycles)[peak, i])

        for i in range(0, len(parameters)):
            plt.plot(t[:], np.asarray(sensitivitiesT_2cycles)[:, i], label=parameters[i])

        plt.xlabel('t')
        plt.ylabel('Sensitivity')
        plt.savefig('images/SensitivityT_2enzymes.pdf')
        plt.show()

        print('Total Sensitivity peaks for 2 cycles:')
        for i in range(0, len(parameters)):
            print('\t', parameters[i], 'peaks:')
            peak_list, _ = find_peaks(np.asarray(sensitivitiesT_2cycles)[100:, i])
            peak_list = np.sort(peak_list)
            for peak in peak_list:
                print('\t\t at t =', t[peak], 'for a sensitivity of', np.asarray(sensitivitiesT_2cycles)[peak, i])

        print(np.asarray(sensitivities1_2cycles)[:, 0])

        for i in range(0, len(parameters)):
            plt.plot(t[:], np.asarray(sensitivities1_2cycles)[:, i] / max(np.asarray(sensitivities1_2cycles)[1:, i]),
                     label=parameters[i])

        plt.legend(loc='center right')
        plt.xlabel('t')
        plt.ylabel('sensitivity')
        plt.yticks([])
        plt.savefig('images/SensitivityScaled1_2enzymes.pdf')
        plt.show()

if __name__ == '__main__':
    parameters = [
        # "k_c",
        # "k_e",
        # "k_off",
        # "k_on",
        # "K_d",
        # "Q",
        # "D",
        # "Au",
        # "R_T",
        # "S",
        # "Da",
        # "k_on_D",
        # "k_off_D",
        # "B",
        # "k_plus",
        # "k_minus",
        # "W",
        # "k_c_P",
        # "k_e_P",
        # "k_a_P",
        # "k_abase_P",
        # "nu",
        "G_4",
        # "V_m1",
        # "V_m2",
        # "V_m3",
        # "V_m4",
        # "V_m5",
        # "V_m6",
        # "K_m1",
        # "K_m2",
        # "K_m3",
        # "K_m4",
        # "K_m5",
        # "K_m6",
        "amplitude",
        "G_1",
        "G_2",
        "G_3"
    ]

    exponential = 2

    bounds = [[-exponential, exponential]] * len(parameters)

    parameter_base_values = [
        # 0.2,  # k_c
        # 0.1,  # k_e
        # 0.1,  # k_off
        # 100000000.0,  # k_on
        # 1e-09,  # K_d
        # 3000.0,  # Q
        # 1e-06,  # D
        # 0.026428300073792255,  # Au
        # 50000.0,  # R_T
        # 10000.0,  # S
        # 0.04404716678965375,  # Da
        # 0.1  # k_on_D
        # 10000000.0  # k_off_D
        # 1e-07  # B
        # 4.166666666666667e-11  # k_plus
        # 41666.66666666667  # k_minus
        # 1.000000000000001,  # W
        # 0.02,  # k_c_P
        # 0.1,  # k_e_P
        # 0.5,  # k_a_P
        # 0.001,  # k_abase_P
        # 5.0,  # nu
        1,  # G_4
        # 0.5,  # V_m1
        # 0.15,  # V_m2
        # 0.15,  # V_m3
        # 0.15,  # V_m4
        # 0.25,  # V_m5
        # 0.05,  # V_m6
        # 0.2,  # K_m1
        # 0.2,  # K_m2
        # 0.2,  # K_m3
        # 0.2,  # K_m4
        # 0.2,  # K_m5
        # 0.2,  # K_m6
        2.5,  # amplitude
        0.2775,  # G_1
        0.0005,  # G_2
        1  # G_3
    ]

    cycle_3_sensitivity(parameters, exponential, parameter_base_values, cycles=[2,3,4])
