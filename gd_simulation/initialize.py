import random

import math
import numpy as np
from global_variables import N
from model import Model

modeNorm = pow(N, (3. / 2.)) / math.sqrt(2)

class Initialize():
    @staticmethod
    def _set_mode(i, l, k, m2, real=0):
        momentum2 = pow(2 * math.pi, 2) * (pow(min(i, N - i), 2) + pow(min(l, N - l), 2) + pow(min(k, N - k), 2)) / pow(N,
                                                                                                                        2.)  # p^2 = (2*Pi/L)^2*(i^2 + j^2 + k^2)     L=N (if h=1, which we assume)
        if momentum2 + m2 > 0:
            omega = math.sqrt(momentum2 + m2)
        else:
            omega = 0
        if omega > 0:
            amplval = modeNorm / math.sqrt(2 * omega)
        else:
            amplval = 0
        # left mode
        ampl = amplval * math.sqrt(math.log(1. / random.random()))
        phase = 2. * math.pi * random.random()
        re_f_left = ampl * math.cos(phase)
        im_f_left = ampl * math.sin(phase)
        ampl = amplval * math.sqrt(math.log(1. / random.random()))
        phase = 2. * math.pi * random.random()
        re_f_right = ampl * math.cos(phase)
        im_f_right = ampl * math.sin(phase)

        result = []
        if real == 0:
            result.append((re_f_left + re_f_right) + (im_f_left + im_f_right) * 1j)
            result.append(omega * (im_f_left - im_f_right) - (-omega * (re_f_left - re_f_right)) * 1j)
        else:
            result.append(re_f_left + re_f_right)
            result.append(omega * (im_f_left - im_f_right))
        return result

    @staticmethod
    def initialize_field_non_zero_modes(m2):
        Modes = [[[0. for i in range(N // 2 + 1)] for j in range(N)] for k in
                 range(N)]
        vModes = [[[0. for i in range(N // 2 + 1)] for j in range(N)] for k in range(N)]
        for i in range(N):
            if i == 0:
                iconj = 0
            else:
                iconj = N - i
            for j in range(N):
                for k in range(1, N // 2):
                    Modes[i][j][k], vModes[i][j][k] = Initialize._set_mode(i, j, k,
                                                              m2)
                if j > N // 2 or (i > N // 2 and (j == 0 or j == N // 2)):
                    if j == 0:
                        jconj = 0
                    else:
                        jconj = N - j
                    Modes[i][j][0], vModes[i][j][0] = Initialize._set_mode(i, j, 0, m2)
                    Modes[iconj][jconj][0] = np.conjugate(Modes[i][j][0])
                    vModes[iconj][jconj][0] = np.conjugate(vModes[i][j][0])
                    Modes[i][j][N // 2], vModes[i][j][N // 2] = Initialize._set_mode(i, j, N // 2, m2)
                    Modes[iconj][jconj][N // 2] = np.conjugate(Modes[i][j][N // 2])
                    vModes[iconj][jconj][N // 2] = np.conjugate(vModes[i][j][N // 2])
                elif (i == 0 or i == N // 2) and (j == 0 or j == N // 2):
                    Modes[i][j][0], vModes[i][j][0] = Initialize._set_mode(i, j, 0, m2, 1)
                    Modes[i][j][N // 2], vModes[i][j][N // 2] = Initialize._set_mode(i, j, N // 2, m2, 1)
        Modes[0][0][0] = 0.
        vModes[0][0][0] = 0.
        return [np.fft.irfftn(Modes), np.fft.irfftn(vModes)]

    @staticmethod
    def mod_chi_pi_ini(chiPi, chi):
        return chiPi * Model.non_canonical_multiplier(chi)
