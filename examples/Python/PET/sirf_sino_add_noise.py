"""Add noise to a sinogram

Usage:
  sino_to_image [--help | options]

Options:
  -s <file>, --in_sino=<file>   input sinogram
  -o <path>, --out_sino=<path>  output sinogram (default: <input>_<percentage>)
  -p <int>,  --per=<int>        percentage of counts to keep
"""

# CCP PETMR Synergistic Image Reconstruction Framework (SIRF)
# Copyright 2020 University College London.
#
# This is software developed for the Collaborative Computational
# Project in Positron Emission Tomography and Magnetic Resonance imaging
# (http://www.ccppetmr.ac.uk/).
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from sirf.Utilities import error
import sirf.STIR as pet
import numpy as np
from docopt import docopt

__version__ = '0.1.0'
args = docopt(__doc__, version=__version__)

# Get filenames
f_in_sino = args['--sino']
f_out_sino = args['--img']
percentage_counts = int(args['--per'])


def add_sino_noise(fraction_of_counts, sinogram):
    sino_arr = sinogram.as_array()
    minmax = (sino_arr.min(), sino_arr.max())
    if 0 < fraction_of_counts <= 1:
        counts = fraction_of_counts * (minmax[1] - minmax[0])
    elif isinstance(fraction_of_counts, int):
        pass

    sino_arr = counts * ((sino_arr - minmax[0]) / (minmax[1] - minmax[0]))
    noisy_counts = sinogram * 0.
    noisy_counts.fill(np.random.poisson(sino_arr))

    return noisy_counts


def main():
    sino = pet.AcquisitionData(f_in_sino)
    sino = add_sino_noise(float(percentage_counts)/100.)
    sino.write(f_out_sino + "_" + str(percentage_counts) + "-percent")


# if anything goes wrong, an exception will be thrown 
# (cf. Error Handling section in the spec)
try:
    main()
    print('done')
except error as err:
    # display error information
    print('%s' % err.value)
