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
This file contains the training function for the quadratic model.
For details and references, see Section III.D in:

K. Bredies, E. Chenchene, A. Hosseini.
A hybrid proximal generalized conditional gradient method and
application to total variation parameter learning, 2022.

Submitted to ECC23, within the EUCA Series of European Control Conferences
To be held in Bucharest, Romania, from June 13 to June 16, 2023.
"""

import numpy as np
import structures as st
from plots import compare_results, plot_D


def D_func_qm(Sig, Sig_old, A, A_old, U, Dx, Dy, M1, M2, Fs, Fs_b, lam, n, N):

    piece_1 = 0
    piece_2 = 0
    piece_1 = 1/N*np.sum(np.multiply(st.div_m(Sig_old, M1, M2)+Fs,
                                         st.div_m(Sig_old, M1, M2)-st.div_m(Sig, M1, M2)))
    piece_2 = 1/N*np.multiply(st.TV(U, Dx, Dy, n, N)[np.newaxis, :], Fs_b)@Fs_b.T
    piece_2 = np.sum(np.multiply(piece_2, A_old-A))

    return piece_1+piece_2-lam/2*np.sum(np.square(A_old-A))


def step_size_qm(Sig, Sig_old, A, A_old, Du, lam, N):
    '''
    The Lipschitz constant can be estimated by 8/N.
    '''
    dist = np.sum(np.square(Sig-Sig_old))+np.sum(np.square(A-A_old))

    return min(1, N*(Du+lam/2*np.sum(np.square(A-A_old)))/(8*dist))


def train_qm(dataset, lam, maxit):

    print("\n *** Starting training quadratic model. Info displayed every 30 iterations. *** \
\n ******* Note that this phase may take several hours *******")
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

    # Add bias term to noisy images for flexibility
    Fs_b = np.vstack([Fs, np.ones(N)])

    # initializing
    Sig = np.zeros((n, N, 2))
    A = np.zeros((n+1, n+1))

    Du_list = []

    D_u = 1
    it = 0
    while D_u > 1e-4 and it <= maxit:

        A_old = np.copy(A)
        Sig_old = np.copy(Sig)

        # partial update on A
        pars = st.TV(U, Dx, Dy, n, N)-st.TV(st.div_m(Sig, M1, M2)+Fs, Dx, Dy, n, N)
        temp = np.multiply(pars[np.newaxis, :], Fs_b)@Fs_b.T
        A = st.proj_pos(A-1/(N*lam)*temp)

        # partial update on Sig
        Sig = st.grad_m(st.div_m(Sig, M1, M2)+Fs, Dx, Dy, n, N)
        Norms = np.linalg.norm(Sig, axis=2)
        Norms = np.repeat(Norms[:, :, np.newaxis], 2, axis=2)
        Sig[Norms > 0] = np.divide(Sig[Norms > 0], Norms[Norms > 0])  # normalized

        for i in range(N):
            Sig[:, i, :] = Fs_b[:, i].T@A@Fs_b[:, i]*Sig[:, i, :]

        # step size
        D_u = D_func_qm(Sig, Sig_old, A, A_old, U, Dx, Dy, M1, M2, Fs, Fs_b, lam, n, N)
        theta = step_size_qm(Sig, Sig_old, A, A_old, D_u, lam, N)

        Du_list.append(D_u)

        # full update on A and Sig
        A = A_old+theta*(A-A_old)
        Sig = Sig_old+theta*(Sig-Sig_old)
        it += 1

        if it % 200 == 100:
            print(f'\n ########## Iteration: {it}')
            print(f'step size = {theta}')
            print(f'D(u^k) = {D_u}')
            print(f'|A(k+1)-A(k)| = {np.linalg.norm(A-A_old)}')
            print('Plotting 5 patches...')
            compare_results(Sig, dataset, int(min(5, N)))
            print('Plotted.')
            # plotting
            plot_D(Du_list)

        if it % 30 == 5 and it % 200 != 5:
            print(f'\n ########## Iteration: {it}')
            print(f'step size = {theta}')
            print(f'D(u^k) = {D_u}')
            print(f'|A(k+1)-A(k)| = {np.linalg.norm(A-A_old)}')

    print(f'\n ########## Final iteration: {it}')
    print(f'Final residual D(u^k) = {D_u}')
    print('Plotting 5 patches...')
    compare_results(Sig, dataset, int(min(5, N)))
    print('Plotting the residual as a function of the iterations...')
    # plotting
    plot_D(Du_list)
    print('Residual as a function of the iteration plotted and saved as du.pdf')

    return A, Sig, Du_list
