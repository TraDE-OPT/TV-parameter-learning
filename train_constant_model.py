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
This file contains the training function for the constant model.
For details and references, see Section III.D in:

K. Bredies, E. Chenchene, A. Hosseini.
A hybrid proximal generalized conditional gradient method and
application to total variation parameter learning, 2022.

Submitted to ECC23, within the EUCA Series of European Control Conferences
To be held in Bucharest, Romania, from June 13 to June 16, 2023.
"""

import numpy as np
import structures as st


def D_func_cm(Sig, Sig_old, a, a_old, U, Dx, Dy, M1, M2, Fs, lam, n, N):

    piece_1 = 0
    piece_2 = 0
    piece_1 = 1/N*np.sum(np.multiply(st.div_m(Sig_old, M1, M2)+Fs,
                                         st.div_m(Sig_old, M1, M2)-st.div_m(Sig, M1, M2)))
    piece_2 = np.mean(st.TV(U, Dx, Dy, n, N))*(a_old-a)

    return piece_1+piece_2-lam/2*(a_old-a)**2


def step_size_cm(Sig, Sig_old, a, a_old, Du, lam, N):
    '''
    The Lipschitz constant can be estimated by 8/N.
    '''
    dist = np.sum(np.square(Sig-Sig_old))+(a-a_old)**2

    return min(1, N*(Du + lam/2*(a-a_old)**2)/(8*dist))


def train_cm(dataset, lam, maxit):

    print("\n *** Starting training constant model. Info displayed every 1500 iterations. *** ")
    # structure
    n = len(dataset[0][0])
    size_patches = int(np.sqrt(n))
    N = len(dataset)  # number of patches

    Dx = st.grad_x(size_patches)
    Dy = st.grad_y(size_patches)
    M1 = -Dx.T
    M2 = -Dy.T

    # true and noisy images as numpy array for speed
    U = np.zeros((n, N))
    Fs = np.zeros((n, N))
    for k in range(N):
        U[:, k] = dataset[k][0]
        Fs[:, k] = dataset[k][1]

    # initializing
    Sig = np.zeros((n, N, 2))
    a = 0

    Du_list = []

    D_u = 1
    it = 0
    while D_u > 1e-5 and it <= maxit:

        a_old = a
        Sig_old = np.copy(Sig)

        # partial update on A
        pars = st.TV(U, Dx, Dy, n, N)-st.TV(st.div_m(Sig, M1, M2)+Fs, Dx, Dy, n, N)
        a = np.maximum(a-np.mean(pars)/lam, 0)

        # partial update on Sig
        Sig = st.grad_m(st.div_m(Sig, M1, M2)+Fs, Dx, Dy, n, N)
        Norms = np.linalg.norm(Sig, axis=2)
        Norms = np.repeat(Norms[:, :, np.newaxis], 2, axis=2)
        Sig[Norms > 0] = np.divide(Sig[Norms > 0], Norms[Norms > 0])  # normalized
        Sig = a*Sig

        # step size
        D_u = D_func_cm(Sig, Sig_old, a, a_old, U, Dx, Dy, M1, M2, Fs, lam, n, N)
        theta = step_size_cm(Sig, Sig_old, a, a_old, D_u, lam, N)

        Du_list.append(D_u)

        # full update on A and Sig
        a = a_old+theta*(a-a_old)
        Sig = Sig_old+theta*(Sig-Sig_old)
        it += 1

        if it % 1500 == 5 and it >= 6:
            print(f'\n ########## Iteration: {it}')
            print(f'Parameter alpha = {a}')
            print(f'step size = {theta}')
            print(f'D(u^k) = {D_u}')
            print(f'|a(k+1)-a(k)| = {np.abs(a-a_old)}')

    print(f'\n ########## Final iteration: {it}')
    print(f'Final parameter alpha = {a}')
    print(f'Final residual D(u^k) = {D_u}')

    return a, Sig, Du_list
