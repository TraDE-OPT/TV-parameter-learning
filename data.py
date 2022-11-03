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
Data processing.
For details and references, see Section III.D in:

K. Bredies, E. Chenchene, A. Hosseini.
A hybrid proximal generalized conditional gradient method and
application to total variation parameter learning, 2022.

Submitted to ECC23, within the EUCA Series of European Control Conferences
To be held in Bucharest, Romania, from June 13 to June 16, 2023.

"""

import glob
import random
import numpy as np
from PIL import Image


def read_image(img_name, px=10, py=10):

    img = Image.open(img_name).convert('L')
    image_arr = 255-np.array(img.resize((px, py)), dtype=np.float64)
    image_arr /= 255

    return image_arr


def apply_noise(image, noise_level):

    px, py = np.shape(image)
    gauss = np.random.normal(0, noise_level, (px, py))

    return image+gauss


def create_sample(noise_level, folder, size_patches=16):

    dataset = []
    num_files = 0

    for filename in glob.glob(f'Figures/Cartoon/{folder}/*'):

        img = Image.open(filename).convert('L')
        sxx, syy = np.shape(img)
        sx = int(((sxx/2.3)//size_patches)*size_patches)
        sy = int(((syy/2.3)//size_patches)*size_patches)
        image_arr = 255-np.array(img.resize((sx, sy)), dtype=np.float64)

        image_arr /= 255

        for i in np.arange(0, sy, size_patches):
            for j in np.arange(0, sx, size_patches):

                patch = image_arr[i:i+size_patches, j:j+size_patches]
                gauss = np.random.normal(0, noise_level, (size_patches, size_patches))

                dataset.append([patch.reshape(int(size_patches**2)),
                                    (patch+gauss).reshape(int(size_patches**2))])

        num_files += 1

    print(f"Dataset has size: {len(dataset)}\nNumber of pictures: {num_files}")
    return random.sample(dataset, len(dataset))  # shuffle for better visualization
