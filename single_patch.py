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
This file contains the function that allows us to find the best parameter for one figure.
For details and references, see Section III.D in:

K. Bredies, E. Chenchene, A. Hosseini.
A hybrid proximal generalized conditional gradient method and
application to total variation parameter learning, 2022.

Submitted to ECC23, within the EUCA Series of European Control Conferences
To be held in Bucharest, Romania, from June 13 to June 16, 2023.
"""

import numpy as np
import structures as st


def D_func_single_patch(sig, sig_old, a, a_old, u, Dx, Dy, M1, M2, f, lam, n):

    piece_1 = 0
    piece_2 = 0
    piece_1 = np.sum(np.multiply(st.div(sig_old, M1, M2)+f,
                                     st.div(sig_old, M1, M2)-st.div(sig, M1, M2)))
    piece_2 = st.tv(u, Dx, Dy, n)*(a_old-a)

    return piece_1+piece_2-lam/2*(a-a_old)**2


def step_size_single_patch(sig, sig_old, a, a_old, Du, lam):
    '''
    The Lipschitz constant can be estimated by 8.
    '''
    dist = np.sum(np.square(sig-sig_old))+(a-a_old)**2

    return min(1, (Du + lam/2*(a_old-a)**2)/(8*dist))


def single_patch_best_par(true_image, noisy_image, lam, maxit, show_details=True):

    # structure
    n = len(noisy_image)
    size_patches = int(np.sqrt(n))

    Dx = st.grad_x(size_patches)
    Dy = st.grad_y(size_patches)
    M1 = -Dx.T
    M2 = -Dy.T

    u = np.copy(true_image)
    f = np.copy(noisy_image)

    # initializing
    sig = np.zeros((n, 2))
    a = 0

    Du_list = []

    for it in range(maxit):

        a_old = a
        sig_old = np.copy(sig)

        # partial update on A
        par = st.tv(u, Dx, Dy, n)-st.tv(st.div(sig, M1, M2)+f, Dx, Dy, n)
        a = np.maximum(a-par/lam, 0)

        # partial update on Sig
        sig = st.grad(st.div(sig, M1, M2)+f, Dx, Dy, n)
        Norms = np.linalg.norm(sig, axis=1)
        Norms = np.repeat(Norms[:, np.newaxis], 2, axis=1)
        sig[Norms > 0] = np.divide(sig[Norms > 0], Norms[Norms > 0])  # normalized
        sig = a*sig

        # step size
        D_u = D_func_single_patch(sig, sig_old, a, a_old, u, Dx, Dy, M1, M2, f, lam, n)
        theta = step_size_single_patch(sig, sig_old, a, a_old, D_u, lam)

        Du_list.append(D_u)

        # full update on A and Sig
        a = a_old+theta*(a-a_old)
        sig = sig_old+theta*(sig-sig_old)

        if show_details:

            if it % 10000 == 5:
                print(f'\n ########## Iteration: {it}')
                print(f"best parameter = {a}")
                print(f'step size = {theta}')
                print(f'D(u^k) = {D_u}')
                print(f'|a(k+1)-a(k)| = {np.abs(a-a_old)}')

    return a, sig, Du_list
