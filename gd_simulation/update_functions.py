#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 08:34:50 2018

@author: michal
"""
import numpy as np
from global_variables import N
from model import Model
from utils import pw2


def chiPiFirstUpdate(h, a, phiPi, chi):
    return (h / (2. * pw2(a))) * pw2(phiPi) * Model.d_non_canonical_multiplier(chi) / (
                2 * pw2(Model.non_canonical_multiplier(chi)))


def phiFirstUpdate(h, a, phiPi, chi):
    return (h / (2. * pw2(a))) * phiPi / Model.non_canonical_multiplier(chi)


def paFirstUpdate(h, phiPi, chi, a):
    return pow(N, 3) * (h / 2.) * np.average(pw2(phiPi) / Model.non_canonical_multiplier(chi)) / pow(a, 3)


def chiPiPartSecondUpdate(chi):
    return Model.d_non_canonical_multiplier(chi) / 2.


def _dgrad2(field):
    return 12. * field - 2. * (
                np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) + np.roll(field, 1, axis=1) + np.roll(field, -1,
                                                                                                             axis=1) + np.roll(
            field, 1, axis=2) + np.roll(field, -1, axis=2))


def _dgradPhi_non_canonical_kinetic_part(chi):
    return Model.non_canonical_multiplier(chi) / 2. + (
                Model.non_canonical_multiplier(np.roll(chi, 1, axis=0)) + Model.non_canonical_multiplier(
            np.roll(chi, -1, axis=0)) + Model.non_canonical_multiplier(
            np.roll(chi, 1, axis=1)) + Model.non_canonical_multiplier(
            np.roll(chi, -1, axis=1)) + Model.non_canonical_multiplier(
            np.roll(chi, 1, axis=2)) + Model.non_canonical_multiplier(np.roll(chi, -1, axis=2))) / 12.


def dgradChiTerm(chi):
    return _dgrad2(chi)


def dgradPhiTerm(phi, chi):
    return _dgrad2(phi) * _dgradPhi_non_canonical_kinetic_part(chi)
