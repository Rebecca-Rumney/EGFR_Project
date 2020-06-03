import numpy as np
import scipy.constants
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import ODE_solver as ode

def steady_state_coupled(start_point, cycles=3, params=None):
    equations = ode.ODESolving(params={**params, 'amplitude': 0})
    def function(A_V):
        if cycles == 3:
            return equations.coupled3_dydt(1000, A_V)
        if cycles == 4:
            return equations.coupled4_dydt(1000, A_V)
        if cycles == 2:
            return equations.coupled2_dydt(1000, A_V)
    sol = fsolve(function, start_point)
    return sol


def plot_line(G_values, cycles=3):
    sensible_params = {'amplitude': 0, 'G_1': 0.2775, 'R_T': 5e4}
    last_index = cycles+3
    equation = 'coupled' + str(cycles)

    for i in range(0, len(G_values)):
        plt.subplot(2, 2, i+1)
        list_of_steady_states = []
        parameter_name = 'G_' + str(i+1)
        for value in G_values[i]:
            initial_guess = ode.ODESolving({**sensible_params, parameter_name: value}).solve(equations=equation).y[:, -1]
            steady_state = steady_state_coupled(initial_guess, cycles=cycles, params={**sensible_params, parameter_name: value})
            list_of_steady_states.append(steady_state)

        steady_states = np.asarray(list_of_steady_states)

        if i == 3:
            x_marks = np.power(10, exponents_G4)
            line = plt.plot(x_marks, steady_states[:, last_index])
            lines.append(line[0])
        else:
            x_marks = np.power(10, exponents)
            plt.plot(x_marks, steady_states[:, last_index])


if __name__ == '__main__':
    # plt.figure(1)
    # plt.figure(2)
    # plt.figure(3)
    # plt.figure(4)

    lines = []

    exponents = np.linspace(-2, 2, 101)
    exponents_G4 = np.linspace(-10, 10, 101)
    G1_values = 0.2775 * np.power(10, exponents)
    G2_values = 5e-4 * np.power(10, exponents)
    G3_values = 1 * np.power(10, exponents)
    G4_values = 1 * np.power(10, exponents_G4)

    file_name = 'images/enzymes_steady_state'

    plot_line([G1_values, G2_values, G3_values, G4_values], cycles=2)
    plot_line([G1_values, G2_values, G3_values, G4_values], cycles=3)
    plot_line([G1_values, G2_values, G3_values, G4_values], cycles=4)

    for i in range(0, 4):
        plt.subplot(2, 2, i+1)
        plt.ylabel('Output')
        plt.xlabel('G_' + str(i+1) + ' Multiplier')
        plt.xscale('log')
        if i == 3:
            plt.yscale('linear')
        plt.grid()
    plt.savefig(file_name + 'G1to4.pdf')
    plt.show()

    cycles = 3
    sensible_params = {'amplitude': 0, 'G_1': 0.2775, 'R_T': 5e4}
    last_index = cycles+3
    equation = 'coupled' + str(cycles)

    plt.subplot(2, 2, 3)
    list_of_steady_states = []
    parameter_name = 'G_4'
    for value in G4_values:
        initial_guess = ode.ODESolving({**sensible_params, parameter_name: value}).solve(equations=equation).y[:, -1]
        steady_state = steady_state_coupled(initial_guess, cycles=cycles,
                                            params={**sensible_params, parameter_name: value})
        list_of_steady_states.append(steady_state)

    steady_states = np.asarray(list_of_steady_states)
    x_marks = np.power(10, exponents_G4)
    plt.plot(x_marks, steady_states[:, last_index], color='C1')

    cycles = 4
    sensible_params = {'amplitude': 0, 'G_1': 0.2775, 'R_T': 5e4}
    last_index = cycles + 3
    equation = 'coupled' + str(cycles)

    list_of_steady_states = []
    parameter_name = 'G_4'
    for value in G4_values:
        initial_guess = ode.ODESolving({**sensible_params, parameter_name: value}).solve(equations=equation).y[:, -1]
        steady_state = steady_state_coupled(initial_guess, cycles=cycles,
                                            params={**sensible_params, parameter_name: value})
        list_of_steady_states.append(steady_state)

    steady_states = np.asarray(list_of_steady_states)
    x_marks = np.power(10, exponents_G4)
    plt.plot(x_marks, steady_states[:, last_index], color='C2')

    plt.ylabel('Output')
    plt.xlabel('G_' + str(4) + ' Multiplier')
    plt.xscale('log')
    plt.grid()
    plt.figlegend(lines, ('2 enzyme cycles', '3 enzyme cycles', '4 enzyme cycles'), loc='upper left')
    plt.savefig(file_name + 'G4reduced.pdf')
    plt.show()
