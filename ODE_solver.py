import numpy as np
import scipy.constants
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class ODESolving:

    def __init__(self, params=None, verbose=False):
        # **Ligand Binding**
        # Units
        if params is None:
            params = {}
        minute = 1
        molecules = 1
        cm = 1
        mole = scipy.constants.N_A * molecules
        litre = 1000 * cm ** 3
        M = mole / litre
        r_cell = 5e-4  # cm
        A = 2 * np.pi * (r_cell ** 2)  # cm^2
        cell = A
        second = minute / 60

        # Parameter values
        self.params = {}
        self.units = {}
        self.params['k_c'], self.units['k_c'] = 0.2, "1/minute"
        self.params['k_e'], self.units['k_e'] = 0.1, "1/minute"

        self.params['k_off'], self.units['k_off'] = 0.1, "1 / minute"
        self.params['k_on'], self.units['k_on'] = 1e8, "1 / (minute * M)"
        self.params['K_d'], self.units['K_d'] = None, "M"

        self.params['Q'], self.units['Q'] = 3e3, "molecules / minute / cell"
        self.params['D'], self.units['D'] = 1e-6, "cm ** 2 / second"
        self.params['Au'], self.units['Au'] = None, ""

        self.params['R_T'], self.units['R_T'] = 5e4, "molecules /cell"
        self.params['S'], self.units['S'] = None, "molecules /cell /minute"
        self.params['Da'], self.units['Da'] = None, ""

        self.params['k_on_D'], self.units['k_on_D'] = 0.1, "1/(M*minute)"
        self.params['k_off_D'], self.units['k_off_D'] = 1e7, "1/minute"
        self.params['B'], self.units['B'] = 1e-7, "M"
        self.params['k_plus'], self.units['k_plus'] = None, ""
        self.params['k_minus'], self.units['k_minus'] = None, ""
        self.params['W'], self.units['W'] = None, ""

        # **Protease activation**
        self.params['k_c_P'], self.units['k_c_P'] = 0.02, "1/minute"
        self.params['k_e_P'], self.units['k_e_P'] = 0.1, "1/minute"
        self.params['k_a_P'], self.units['k_a_P'] = 0.5, "1/minute"
        self.params['k_abase_P'], self.units['k_abase_P'] = 1e-3, "1/minute"
        self.params['nu'], self.units['nu'] = None, ""

        # **Signalling cascade**
        self.params['G_4'], self.units['G_4'] = 1, ""
        self.params['V_m1'], self.units['V_m1'] = 0.5, ""  # or 1?
        self.params['V_m2'], self.units['V_m2'] = 0.15, ""
        self.params['V_m3'], self.units['V_m3'] = 0.15, ""
        self.params['V_m4'], self.units['V_m4'] = 0.15, ""
        self.params['V_m5'], self.units['V_m5'] = 0.25, ""
        self.params['V_m6'], self.units['V_m6'] = 0.05, ""

        K_mj = [0.2] * 6
        K_mj_units = [""] * 6
        self.params['K_m1'], self.params['K_m2'], self.params['K_m3'], self.params['K_m4'], self.params['K_m5'], \
        self.params['K_m6'] = K_mj
        self.units['K_m1'], self.units['K_m2'], self.units['K_m3'], self.units['K_m4'], self.units['K_m5'], self.units[
            'K_m6'] = K_mj_units

        self.params['amplitude'], self.units['amplitude'] = 2.5, ""

        # **Coupled**
        self.params['G_1'], self.units['G_1'] = 0.3, ""
        self.params['G_2'], self.units['G_2'] = 5e-4, ""
        self.params['G_3'], self.units['G_3'] = 1, ""

        for parameter in params:
            if parameter in self.params:
                self.params[parameter] = params[parameter]
            else:
                print(parameter + " is not used calculations. Please check spelling.")

        if 'K_d' not in params:
            self.params['K_d'] = self.params['k_off'] / self.params['k_on']
        if 'Au' not in params:
            self.params['Au'] = (self.params['Q'] * (molecules / minute / cell) * r_cell * cm) / (
                    self.params['D'] * (cm ** 2 / second) * self.params['K_d'] * M)
        if 'S' not in params:
            self.params['S'] = self.params['R_T'] * self.params['k_c']
        if 'Da' not in params:
            self.params['Da'] = (self.params['k_on'] * (1 / (minute * M)) * r_cell * (cm) * self.params['S'] * (
                    molecules / (cell * minute))) / (
                                        self.params['k_c'] * (1 / minute) * self.params['D'] * (cm ** 2 / second))
        if 'k_plus' not in params:
            self.params['k_plus'] = (self.params['k_on_D'] * (1 / (M * minute)) * self.params['B'] * (M) * (
                    r_cell * (cm)) ** 2) / (self.params['D'] * (cm ** 2 / second))
        if 'k_minus' not in params:
            self.params['k_minus'] = (self.params['k_off_D'] * (1 / minute) * (r_cell * cm) ** 2) / (
                    self.params['D'] * (cm ** 2 / second))
        if 'W' not in params:
            self.params['W'] = (1 + np.power(self.params['k_plus'] + self.params['k_minus'], 0.5)) / (
                    1 + (self.params['k_minus'] / np.power(self.params['k_plus'] + self.params['k_minus'], 0.5)))
        if 'nu' not in params:
            self.params['nu'] = self.params['k_a_P'] / self.params['k_e_P']
        #
        # if verbose:
        #     for parameter in self.params:
        #         print(parameter, "=", self.params[parameter], self.units[parameter])

        if verbose:
            print('[')
            for parameter in self.params:
                print('[0,  upper_multiplier *', self.params[parameter], ']  # ', parameter)
            print(']')
        # if verbose:
        #     print('[')
        #     for parameter in self.params:
        #         print('"' + parameter + '", ')
        #     print(']')

    def increasing_input(self, t):
        if t < 20:
            return 0
        else:
            return (t - 20) * (self.params['amplitude'] / 160.0)

    def pulse_input(self, t):
        if 0.0 <= t <= 2.0:
            return self.params['amplitude']
        else:
            return 0

    def steady_input(self, t):
        return self.params['amplitude']

    def receptor_dydt(self, t, R_V, Au):
        R = R_V[0]
        C = R_V[1]
        dot_R = -self.params['k_off'] * R * (Au + self.params['Da'] * C) / (self.params['W'] + self.params['Da'] * R) + \
                self.params['k_c'] * (1 - R) + self.params['k_off'] * C
        dot_C = self.params['k_off'] * R * (Au + self.params['Da'] * C) / (self.params['W'] + self.params['Da'] * R) - (
                self.params['k_e'] + self.params['k_off']) * C
        return [dot_R, dot_C]

    def protease_dydt(self, t, P_V, nu):
        P = P_V[0]
        P_a = P_V[1]
        dot_P = self.params['k_c_P'] * (1 - P) - self.params['k_e_P'] * nu * P
        dot_P_a = self.params['k_e_P'] * nu * P - self.params['k_e_P'] * P_a
        return [dot_P, dot_P_a]

    def cascade2_dydt(self, t, E_V, I):
        e1p = E_V[0]
        e2p = E_V[1]
        dot_e1p = (I(t) / (1 + self.params['G_4'] * e2p)) * self.params['V_m1'] * (1 - e1p) / (
                self.params['K_m1'] + 1 - e1p) - (self.params['V_m2'] * e1p) / (self.params['K_m2'] + e1p)
        dot_e2p = self.params['V_m5'] * e1p * (1 - e2p) / (self.params['K_m5'] + 1 - e2p) - self.params[
            'V_m6'] * e2p / (self.params['K_m6'] + e2p)
        return [dot_e1p, dot_e2p]

    def cascade3_dydt(self, t, E_V, I):
        e1p = E_V[0]
        e2p = E_V[1]
        e3p = E_V[2]
        dot_e1p = (I(t) / (1 + self.params['G_4'] * e3p)) * self.params['V_m1'] * (1 - e1p) / (
                self.params['K_m1'] + 1 - e1p) - (self.params['V_m2'] * e1p) / (self.params['K_m2'] + e1p)
        dot_e2p = self.params['V_m3'] * e1p * (1 - e2p) / (self.params['K_m3'] + 1 - e2p) - self.params[
            'V_m4'] * e2p / (self.params['K_m4'] + e2p)
        dot_e3p = self.params['V_m5'] * e2p * (1 - e3p) / (self.params['K_m5'] + 1 - e3p) - self.params[
            'V_m6'] * e3p / (self.params['K_m6'] + e3p)
        return [dot_e1p, dot_e2p, dot_e3p]

    def cascade4_dydt(self, t, E_V, I):
        e1p = E_V[0]
        e2p = E_V[1]
        e3p = E_V[2]
        e4p = E_V[3]
        dot_e1p = (I(t) / (1 + self.params['G_4'] * e4p)) * self.params['V_m1'] * (1 - e1p) / (
                self.params['K_m1'] + 1 - e1p) - (self.params['V_m2'] * e1p) / (self.params['K_m2'] + e1p)
        dot_e2p = self.params['V_m3'] * e1p * (1 - e2p) / (self.params['K_m3'] + 1 - e2p) - self.params[
            'V_m4'] * e2p / (self.params['K_m4'] + e2p)
        dot_e3p = self.params['V_m3'] * e2p * (1 - e3p) / (self.params['K_m3'] + 1 - e3p) - self.params[
            'V_m4'] * e3p / (self.params['K_m4'] + e3p)
        dot_e4p = self.params['V_m5'] * e3p * (1 - e4p) / (self.params['K_m5'] + 1 - e4p) - self.params[
            'V_m6'] * e4p / (self.params['K_m6'] + e4p)
        return [dot_e1p, dot_e2p, dot_e3p, dot_e4p]

    def coupled2_dydt(self, t, A_V, function='pulse_input'):
        R = A_V[0]
        C = A_V[1]
        P = A_V[2]
        P_a = A_V[3]
        e1p = A_V[4]
        e2p = A_V[5]
        Au = self.params['G_1'] * P_a
        nu = self.params['k_abase_P'] / self.params['k_e_P'] + self.params['G_3'] * e2p

        # print('Au =', Au)
        # print('nu =', nu)

        def I(t):
            I = getattr(self, function)(t) + self.params['G_2'] * self.params['R_T'] * C
            return I

        dotR_C = np.asarray(self.receptor_dydt(t, [R, C], Au))
        dotP_Pa = np.asarray(self.protease_dydt(t, [P, P_a], nu))
        dotE = np.asarray(self.cascade2_dydt(t, [e1p, e2p], I))
        dot_all = np.concatenate((dotR_C, dotP_Pa, dotE))
        return dot_all

    def coupled3_dydt(self, t, A_V, function='pulse_input'):
        R = A_V[0]
        C = A_V[1]
        P = A_V[2]
        P_a = A_V[3]
        e1p = A_V[4]
        e2p = A_V[5]
        e3p = A_V[6]
        Au = self.params['G_1'] * P_a
        nu = self.params['k_abase_P'] / self.params['k_e_P'] + self.params['G_3'] * e3p

        # print('Au =', Au)
        # print('nu =', nu)

        def I(t):
            I = getattr(self, function)(t) + self.params['G_2'] * self.params['R_T'] * C
            return I

        dotR_C = np.asarray(self.receptor_dydt(t, [R, C], Au))
        dotP_Pa = np.asarray(self.protease_dydt(t, [P, P_a], nu))
        dotE = np.asarray(self.cascade3_dydt(t, [e1p, e2p, e3p], I))
        dot_all = np.concatenate((dotR_C, dotP_Pa, dotE))
        return dot_all

    def coupled4_dydt(self, t, A_V, function='pulse_input'):
        R = A_V[0]
        C = A_V[1]
        P = A_V[2]
        P_a = A_V[3]
        e1p = A_V[4]
        e2p = A_V[5]
        e3p = A_V[6]
        e4p = A_V[7]
        Au = self.params['G_1'] * P_a
        nu = self.params['k_abase_P'] / self.params['k_e_P'] + self.params['G_3'] * e4p

        # print('Au =', Au)
        # print('nu =', nu)

        def I(t):
            I = getattr(self, function)(t) + self.params['G_2'] * self.params['R_T'] * C
            return I

        dotR_C = np.asarray(self.receptor_dydt(t, [R, C], Au))
        dotP_Pa = np.asarray(self.protease_dydt(t, [P, P_a], nu))
        dotE = np.asarray(self.cascade4_dydt(t, [e1p, e2p, e3p, e4p], I))
        dot_all = np.concatenate((dotR_C, dotP_Pa, dotE))
        return dot_all

    def solve(self, equations='coupled3', verbose=False, cascade_input='pulse_input', times=None):
        if equations == 'coupled2':
            if verbose:
                print('Solving coupled equations')
            V_0 = [1, 0, 1, 0, 0, 0]
            sol = solve_ivp(self.coupled2_dydt, (0, 180), V_0, args=(cascade_input,), method='LSODA', t_eval=times)
            if verbose:
                print('Solved coupled equations with steady state:', sol.y[:, -1])
            return sol

        if equations == 'coupled3':
            if verbose:
                print('Solving coupled equations')
            V_0 = [1, 0, 1, 0, 0, 0, 0]
            sol = solve_ivp(self.coupled3_dydt, (0, 180), V_0, args=(cascade_input,), method='LSODA', t_eval=times)
            if verbose:
                print('Solved coupled equations with steady state:', sol.y[:, -1])
            return sol

        if equations == 'coupled4':
            if verbose:
                print('Solving coupled equations')
            V_0 = [1, 0, 1, 0, 0, 0, 0, 0]
            sol = solve_ivp(self.coupled4_dydt, (0, 180), V_0, args=(cascade_input,), method='LSODA', t_eval=times)
            if verbose:
                print('Solved coupled equations with steady state:', sol.y[:, -1])
            return sol

        if equations == 'receptor':
            if verbose:
                print('Solving receptor equations')
            V_0 = [1, 0]
            sol = solve_ivp(self.receptor_dydt, (0, 180), V_0, args=(self.params['Au'],), method='LSODA', t_eval=times)
            if verbose:
                print('Solved receptor equations with steady state:', sol.y[:, -1])
            return sol

        if equations == 'protease':
            if verbose:
                print('Solving protease equations')
            V_0 = [1, 0]
            sol = solve_ivp(self.protease_dydt, (0, 180), V_0, args=(self.params['nu'],), method='LSODA', t_eval=times)
            if verbose:
                print('Solved protease equations with steady state:', sol.y[:, -1])
            return sol

        if equations == 'cascade3':
            if verbose:
                print('Solving signalling cascade equations with 3 enzymes')
            V_0 = [0, 0, 0]
            sol = solve_ivp(self.cascade3_dydt, (0, 180), V_0, args=(getattr(self, cascade_input),), method='LSODA',
                            t_eval=times)
            if verbose:
                print('Solved signalling cascade equations with steady state:', sol.y[:, -1])
            return sol

        if equations == 'cascade4':
            if verbose:
                print('Solving signalling cascade equations with 3 enzymes')
            V_0 = [0, 0, 0, 0]
            sol = solve_ivp(self.cascade4_dydt, (0, 180), V_0, args=(getattr(self, cascade_input),), method='LSODA',
                            t_eval=times)
            if verbose:
                print('Solved signalling cascade equations with steady state:', sol.y[:, -1])
            return sol

        if equations == 'cascade2':
            if verbose:
                print('Solving signalling cascade equations with 2 enzymes')
            V_0 = [0, 0]
            sol = solve_ivp(self.cascade2_dydt, (0, 180), V_0, args=(getattr(self, cascade_input),), method='LSODA',
                            t_eval=times)
            if verbose:
                print('Solved signalling cascade equations with steady state:', sol.y[:, -1])
            return sol


def plot_figs(cycles=3):

    sensible_params = {'amplitude': 2.5, 'G_1': 0.2775, 'R_T': 5e4}
    variable_labels = ['R bar', 'C bar', 'P bar', 'P_a bar', 'e1p', 'e2p', 'e3p', 'e4p']
    last_index = cycles+3
    equation = 'coupled' + str(cycles)

    # ode_to_graph = ODESolving(params=sensible_params, verbose=True)
    # sol = ode_to_graph.solve(equations=equation)
    # for i in range(0, last_index+1):
    #     plt.plot(sol.t[:], sol.y[i, :], label=variable_labels[i])
    # plt.legend(loc='best')
    # plt.xlabel('t')
    # plt.grid()
    # plt.show()

    exponents = np.linspace(-2, 2, 5)
    file_name = 'images/' + str(cycles) + 'enzymes_'

    pulse = ODESolving({**sensible_params, 'amplitude': 1})
    plt.subplot(221)
    time = [-0.00000000001, 0.0, 0.00000000001, 1.99999999999, 2.0, 2.0000000001, 5.0]
    pulse_values = []
    for t in time:
        pulse_values.append(pulse.pulse_input(t))
    plt.plot(time, pulse_values)
    plt.ylabel('I0(t)')
    plt.yticks([1.0], " ")
    plt.xlabel('time (min)')


    plt.subplot(224)
    amplitude_values = [1, 1.5, 2, 3.5]
    for value in amplitude_values:
        alter_I = ODESolving({**sensible_params, 'amplitude': value})
        sole = alter_I.solve(equations=equation)
        label = 'I_0=' + str(round(value, 4))
        plt.plot(sole.t[:], sole.y[last_index, :], label=label)
    plt.ylabel(variable_labels[last_index])
    plt.legend(loc='best')
    plt.xlabel('time (min)')

    plt.subplot(223)
    G1_values = [0, 0.0775, 0.1775, 0.2775]
    for value in G1_values:
        alter_G = ODESolving({**sensible_params, 'G_1': value})
        sole = alter_G.solve(equations=equation)
        label = 'G_1=' + str(round(value, 4))
        plt.plot(sole.t[:], sole.y[last_index, :], label=label)
    plt.ylabel(variable_labels[last_index])
    plt.legend(loc='best')
    plt.xlabel('time (min)')

    plt.subplot(222)
    RT_values = [3e4, 4e4, 4.5e4, 5e4]
    for value in RT_values:
        alter_RT = ODESolving({**sensible_params, 'R_T': value})
        sole = alter_RT.solve(equations=equation)
        label = 'R_T=' + str(round(value, 4))
        plt.plot(sole.t[:], sole.y[last_index, :], label=label)
    plt.ylabel(variable_labels[last_index])
    plt.legend(loc='best')
    plt.xlabel('time (min)')
    plt.savefig(file_name + 'fig6.pdf')
    plt.show()


    amplitude_values = 2.5 * np.power(10, exponents)
    for value in amplitude_values:
        alter_I = ODESolving({**sensible_params, 'amplitude': value})
        sole = alter_I.solve(equations=equation)
        label = 'I_0=' + str(round(value, 4))
        plt.plot(sole.t[:], sole.y[last_index, :], label=label)
    plt.ylabel(variable_labels[last_index])
    plt.legend(loc='best')
    plt.xlabel('time (min)')
    plt.savefig(file_name + 'I0variation.pdf')
    plt.show()

    G2_values = 5e-4 * np.power(10, exponents)
    for value in G2_values:
        alter_G = ODESolving({**sensible_params, 'G_2': value})
        sole = alter_G.solve(equations=equation)
        label = 'G_2=' + str(round(value, 4))
        plt.plot(sole.t[:], sole.y[last_index, :], label=label)
    plt.ylabel(variable_labels[last_index])
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.savefig(file_name + 'G2variation.pdf')
    plt.show()

    G1_values = 0.2775 * np.power(10, exponents)
    for value in G1_values:
        alter_G = ODESolving({**sensible_params, 'G_1': value})
        sole = alter_G.solve(equations=equation)
        label = 'G_1=' + str(round(value, 4))
        plt.plot(sole.t[:], sole.y[last_index, :], label=label)
    plt.ylabel(variable_labels[last_index])
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.savefig(file_name + 'G1variation.pdf')
    plt.show()

    G3_values= 1 * np.power(10, exponents)
    for value in G3_values:
        alter_G = ODESolving({**sensible_params, 'G_3': value})
        sole = alter_G.solve(equations=equation)
        label = 'G_3=' + str(round(value, 4))
        plt.plot(sole.t[:], sole.y[last_index, :], label=label)
    plt.ylabel(variable_labels[last_index])
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.savefig(file_name + 'G3variation.pdf')
    plt.show()

    G4_values= 1 * np.power(10, exponents)
    for value in G4_values:
        alter_G = ODESolving({**sensible_params, 'G_4': value})
        sole = alter_G.solve(equations=equation)
        label = 'G_4=' + str(round(value, 4))
        plt.plot(sole.t[:], sole.y[last_index, :], label=label)
    plt.ylabel(variable_labels[last_index])
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.savefig(file_name + 'G4variation.pdf')
    plt.show()


if __name__ == '__main__':
    plot_figs(cycles=2)
    plot_figs(cycles=3)
    plot_figs(cycles=4)

    # sol = ode_to_graph.solve(equations='receptor')
    #
    # plt.plot(sol.t[:], sol.y[0, :], label='R bar')
    # plt.plot(sol.t[:], sol.y[1, :], label='C bar')
    # plt.legend(loc='best')
    # plt.xlabel('t')
    # plt.grid()
    # plt.show()
    #
    # solP = ode_to_graph.solve(equations='protease')
    #
    # plt.plot(solP.t[:], solP.y[0, :], label='P')
    # plt.plot(solP.t[:], solP.y[1, :], label='P_a')
    # plt.legend(loc='best')
    # plt.xlabel('t')
    # plt.grid()
    # plt.show()
    #
    # Q_values = [100, 1000, 3000, 5000]
    # for Q in Q_values:
    #     alter_q = ODESolving({**sensible_params, 'Q': Q})
    #     solC = alter_q.solve(equations='receptor')
    #     label = 'Q=' + str(Q)
    #     plt.plot(solC.t[:], solC.y[1, :]*alter_q.params['R_T'], label=label)
    # plt.ylabel('C')
    # plt.legend(loc='best')
    # plt.xlabel('time')
    # plt.show()
    #
    # k_a_P_values = [0.1, 0.2, 0.5, 1.0]
    # plt.figure(1)
    # plt.figure(2)
    # for value in k_a_P_values:
    #     alter_kap = ODESolving(params={**sensible_params, 'k_a_P': value})
    #     solP = alter_kap.solve(equations='protease')
    #     label = 'k_a_P=' + str(value)
    #     plt.figure(1)
    #     plt.plot(solP.t[:], solP.y[1, :], label=label)
    #     plt.figure(2)
    #     plt.plot(solP.t[:], solP.y[1, :] + solP.y[0, :], label=label)
    #
    # plt.figure(2)
    # plt.ylabel('P+P_a')
    # plt.legend(loc='best')
    # plt.xlabel('time')
    #
    # plt.figure(1)
    # plt.ylabel('P_a')
    # plt.legend(loc='best')
    # plt.xlabel('time')
    # plt.show()
    #
    # solE = ode_to_graph.solve(equations='cascade3')
    #
    # plt.plot(solE.t[:], solE.y[0, :], label='e1p')
    # plt.plot(solE.t[:], solE.y[1, :], label='e2p')
    # plt.plot(solE.t[:], solE.y[2, :], label='e3p')
    # plt.legend(loc='best')
    # plt.xlabel('t')
    # plt.grid()
    # plt.show()
    #
    # G_4_values = [0, 1, 2, 4]
    # for value in G_4_values:
    #     alter_G4 = ODESolving(params={**sensible_params, 'G_4': value})
    #     sol_e3p = alter_G4.solve(equations='cascade3', cascade_input='increasing_input')
    #     label = 'G_4=' + str(value)
    #     input_c = (sol_e3p.t[:]-20)*(2.0/120)
    #     plt.plot(input_c, sol_e3p.y[2, :], label=label)
    # plt.ylabel('e_3p')
    # plt.xlabel('input')
    # plt.legend(loc='best')
    # plt.show()
