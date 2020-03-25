"""PET reconstruction using CIL's optimisation algorithms and regularisers

Usage:
  PET_CIL [--help | options]

Options:
  -S <file>, --sino=<file>            sinogram [default: data/examples/PET/mMR/my_forward_projection.hs]
  -i <int>, --iter=<int>              num iterations [default: 10]
  -d <nxny>, --nxny=<nxny>            image x and y dimensions as string '(nx,ny)'
                                      (no space after comma) [default: (127,127)]
  -o <outp>, --outp=<outp>            output file prefix [default: recon]
"""

## CCP PETMR Synergistic Image Reconstruction Framework (SIRF)
## Copyright 2020 University College London.
##
## This is software developed for the Collaborative Computational
## Project in Positron Emission Tomography and Magnetic Resonance imaging
## (http://www.ccppetmr.ac.uk/).
##
## Licensed under the Apache License, Version 2.0 (the "License");
##   you may not use this file except in compliance with the License.
##   You may obtain a copy of the License at
##       http://www.apache.org/licenses/LICENSE-2.0
##   Unless required by applicable law or agreed to in writing, software
##   distributed under the License is distributed on an "AS IS" BASIS,
##   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
##   See the License for the specific language governing permissions and
##   limitations under the License.

import os
from ast import literal_eval
from glob import glob
from docopt import docopt
from sirf.Utilities import error, show_2D_array
import pylab
import sirf.Reg as reg
import sirf.STIR as pet
from ccpi.optimisation.algorithms import PDHG
from ccpi.optimisation.functions import KullbackLeibler, BlockFunction, IndicatorBox
from ccpi.optimisation.operators import CompositionOperator, BlockOperator
from ccpi.plugins.regularisers import FGP_TV
from ast import literal_eval

pet.AcquisitionData.set_storage_scheme('memory')

__version__ = '0.1.0'
args = docopt(__doc__, version=__version__)


def file_exists(filename):
    """Check if file exists, optionally throw error if not"""
    return os.path.isfile(filename)


def check_file_exists(filename):
    """Check file exists, else throw error"""
    if not file_exists:
        raise error('File not found: %s' % filename)


# Sinogram. if sino not found, get the one in the example data
sino_file = args['--sino']
if not file_exists(sino_file):
    sino_file = examples_data_path('PET') + '/mMR/' + sino_file
    check_file_exists(sino_file)

# Number of voxels
nxny = literal_eval(args['--nxny'])
num_iters = int(args['--iter'])

# Output file
outp_prefix = str(args['--outp'])


def main():

    sino = pet.AcquisitionData(sino_file)
    image = acq_data.create_uniform_image(1.0, nxny)

    acq_model = pet.AcquisitionModelUsingRayTracingMatrix()
    acq_model.set_up(sino, image)

    ############################################################################################
    # Set up reconstructor
    ############################################################################################

    print("Setting up reconstructor...")

    # Create composition operators containing acquisition models and resamplers
    # C = [ CompositionOperator(am, res, preallocate=True) for am, res in zip (*(acq_models, resamplers)) ]

    # Configure the PDHG algorithm
    kl = KullbackLeibler(b=sino, eta=(sino * 0 + 1e-5))
    f = BlockFunction(kl)
    K = BlockOperator(acq_model)
    normK = K.norm(iterations=10)

    # normK = LinearOperator.PowerMethod(K, iterations=10)[0]
    # default values
    sigma = 1/normK
    tau = 1/normK 
    sigma = 0.001
    tau = 1/(sigma*normK**2)
    print("Norm of the BlockOperator ", normK)

    if regularisation == 'none':
        G = IndicatorBox(lower=0)
    elif regularisation == 'FGP_TV':
        r_alpha = 5e-1
        r_iterations = 100
        r_tolerance = 1e-7
        r_iso = 0
        r_nonneg = 1
        r_printing = 0
        G = FGP_TV(r_alpha, r_iterations, r_tolerance, r_iso, r_nonneg, r_printing, 'gpu')
    else:
        raise error("Unknown regularisation")

    pdhg = PDHG(f=f, g=G, operator=K, sigma=sigma, tau=tau,
                max_iteration=1000,
                update_objective_interval=1)


    for i in range(1,num_iters+1):
        pdhg.run(1, verbose=True)
        reg.NiftiImageData(pdhg.get_output()).write(outp_prefix + "_iters" + str(i))


# if anything goes wrong, an exception will be thrown 
# (cf. Error Handling section in the spec)
try:
    main()
    print('done')
except error as err:
    # display error information
    print('%s' % err.value)
