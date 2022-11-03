# -*- coding: utf-8 -*-
#
#    Copyright (C) 2022 Kristian Bredies (kristian.bredies@uni-graz.at)
#                       Enis Chenchene (enis.chenchene@uni-graz.at)
#                       Alireza Hosseini (hosseini.alireza@ut.ac.ir)
#
#    This file is part of the example code repository for the paper:
#
#      K. Bredies, E. Chenchene, A. Hosseini.
#      A hybrid proximal generalized conditional gradient method and application
#      to total variation parameter learning, 2022.
#      Submitted to ECC23, within the EUCA Series of European Control Conferences,
#      To be held in Bucharest, Romania, from June 13 to June 16, 2023.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
This file contains several useful functions to reproduce the experiments in Section III.D in:

K. Bredies, E. Chenchene, A. Hosseini.
A hybrid proximal generalized conditional gradient method and
application to total variation parameter learning, 2022.

Submitted to ECC23, within the EUCA Series of European Control Conferences
To be held in Bucharest, Romania, from June 13 to June 16, 2023.
"""

import numpy as np
import scipy.sparse as sp


def grad_x(p):

    diag = np.ones(p)
    diag[-1] = 0
    diag = np.tile(diag, p)

    Dx = sp.spdiags([-diag, [0]+list(diag[:-1])], [0, 1], p**2, p**2)

    return Dx


def grad_y(p):

    diag = np.ones(p**2)
    diag[-p:] = 0*diag[-p:]

    up_diag = np.ones(p**2)
    up_diag[:p] = 0*up_diag[:p]

    Dy = sp.spdiags([-diag, up_diag], [0, p], p**2, p**2)

    return Dy


def grad(psi, Dx, Dy, n):

    sig_out = np.zeros((n, 2))

    sig_out[:, 0] = Dx @ psi
    sig_out[:, 1] = Dy @ psi

    return sig_out


def div(sig, M1, M2):
    '''
    M1 = -Dx* e M2 = -Dy*
    '''
    div_sig_out = M1 @ sig[:, 0] + M2 @ sig[:, 1]

    return div_sig_out


def grad_m(Psi, Dx, Dy, n, N):
    '''
    Computes the gradient of every Psi_i = Psi[:,i]
    '''
    Sig_out = np.zeros((n, N, 2))

    Sig_out[:, :, 0] = Dx @ Psi
    Sig_out[:, :, 1] = Dy @ Psi

    return Sig_out


def div_m(Sig, M1, M2):
    '''
    M1 = -Dx* e M2 = -Dy*
    '''
    DivSig_out = M1 @ Sig[:, :, 0] + M2 @ Sig[:, :, 1]

    return DivSig_out


def tv(psi, Dx, Dy, n):

    return np.sum(np.linalg.norm(grad(psi, Dx, Dy, n), axis=1))


def TV(Psi, Dx, Dy, n, N):

    return np.sum(np.linalg.norm(grad_m(Psi, Dx, Dy, n, N), axis=2), axis=0)


# Proximity operators
def proj_pos(A):

    d, O = np.linalg.eigh(A)

    return O@np.diag(np.maximum(d, 0))@O.T


def prox_fidelity(tau, x, f):

    return (x+tau*f)/(1+tau)


def proj_12(sig):

    sig_out = np.copy(sig)
    norms = np.linalg.norm(sig, axis=1)
    norms = np.repeat(norms[:, np.newaxis], 2, axis=1)
    sig_out[norms > 1] = np.divide(sig_out[norms > 1], norms[norms > 1])

    return sig_out
