"""MCIR for MR

Usage:
  MR_MCIR [--help | options]

Options:
  -T <pattern>, --TMs=<pattern>      transformation matrix pattern (e.g., tm_ms*.txt). Enclose in quotations.
  -k <pattern>, --k_space=<pattern>  k-space pattern (e.g., raw_T1_ms*.h5). Enclose in quotations.
  -o <path>, --output=<path>         output filename [default: recon]
  -i <int>, --iter=<int>             num iterations [default: 10]
  -r <string>, --reg=<string>        regularisation ("none","FGP_TV", ...) [default: none]
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

from glob import glob
from docopt import docopt
from sirf.Utilities import error
import sirf.Reg as reg
import sirf.Gadgetron as mr
from ccpi.optimisation.algorithms import PDHG
from ccpi.optimisation.functions import L2NormSquared, BlockFunction, IndicatorBox, ZeroFunction
from ccpi.optimisation.operators import CompositionOperator, BlockOperator, LinearOperator
from ccpi.plugins.regularisers import FGP_TV



import os
import glob
import numpy as np

__version__ = '0.1.0'
args = docopt(__doc__, version=__version__)

tm_pattern = args['--TMs']
k_space_pattern = args['--k_space']
output_fname = args['--output']
num_iters = int(args['--iter'])
regularisation = args['--reg']



##### TEMPORARY
tm_pattern = "fwd_tm_mf*.txt"
k_space_pattern = "raw_T1_ms*.h5"


def simple_mr_recon(input_data):
    """Simple MR recon from input data"""
    recon = mr.CartesianGRAPPAReconstructor()
    recon.set_input(input_data)
    recon.compute_gfactors(False)
    recon.process()
    return recon.get_output()


def get_resampler_from_tm(tm, image):
    """returns a NiftyResample object for the specified transform matrix and image"""
    resampler = reg.NiftyResample()
    resampler.set_reference_image(image)
    resampler.set_floating_image(image)
    resampler.add_transformation(tm)
    resampler.set_padding_value(0)
    resampler.set_interpolation_type_to_linear()
    return resampler


def save_as_nifti(im, filename):
    """Save an image (e.g., MR) as nii"""
    reg.NiftiImageData(im).write(filename)


def main():

    # ############################################################################################
    # # Parse input files
    # ############################################################################################

    # print("Parsing input arguments...")
    # if tm_pattern is None:
    #     raise AssertionError("--TMs missing")
    # if k_space_pattern is None:
    #     raise AssertionError("--k_space missing")
    # tm_files = sorted(glob(tm_pattern))
    # k_space_files = sorted(glob(k_space_pattern))

    # num_ms = len(tm_files)
    # if num_ms != len(k_space_files):
    #     raise AssertionError("Number of TMs should match number of k-space files")
    # if num_ms == 0:
    #     raise AssertionError("No files found!")

    # ############################################################################################
    # # Read input
    # ############################################################################################

    # print("Reading input...")
    # tms = [0]*num_ms
    # acqs = [0]*num_ms
    # for ind in range(num_ms):
    #     tms[ind] = reg.AffineTransformation(tm_files[ind])
    #     acqs[ind] = mr.AcquisitionData(k_space_files[ind])
    #     # Pre-process data
    #     # acqs[ind] = mr.preprocess_acquisition_data(acqs[ind])

    # ############################################################################################
    # # Set up resamplers
    # ############################################################################################

    # print("Setting up resamplers...")

    # # Need to perform simple recon so we get image shape
    # template_mr_im = simple_mr_recon(acqs[ind])

    # resamplers = [0]*num_ms
    # for ind in range(num_ms):
    #     resamplers[ind] = get_resampler_from_tm(tms[ind], template_mr_im)

    # ############################################################################################
    # # Set up acquisition models
    # ############################################################################################

    # print("Setting up acquisition models...")

    # # Create coil sensitivity data
    # csm = mr.CoilSensitivityData()
    # csm.smoothness = 500
    # csm.calculate(acqs[0])

    # acq_mods = [0]*num_ms
    # for ind in range(num_ms):

    #     # Acquisition model for the individual motion states
    #     acq_mods[ind] = mr.AcquisitionModel(acqs[ind], template_mr_im)
    #     acq_mods[ind].set_coil_sensitivity_maps(csm)











    # Get files
    data_path = os.path.join( '/home/rich/Documents/Data', 'brainweb_single_slice_256')
    transform_matrices_files  = sorted( glob.glob(os.path.join( data_path, 'fwd_tm_mf*.txt') ) )
    T1_template = os.path.join( data_path, 'T1_mf0.nii')



    ############################################################################################
    #  First reconstruct example data to use its metadata, reorient into PET space
    ############################################################################################

    template_acq_data = acq = mr.AcquisitionData(os.path.join(data_path , 'CSM_FULLY_FOV534.h5' ))
    template_acq_data.sort_by_time()
    prep_data = mr.preprocess_acquisition_data(template_acq_data)

    template_mr_im = simple_mr_recon(template_acq_data)
    template_nii = reg.NiftiImageData(T1_template)
    template_mr_im.reorient(template_nii.get_geometrical_info())

    ############################################################################################
    # For each modality and each motion state, resample and replace data in template image
    ############################################################################################

    # Number of motion states
    num_ms = 4

    # Loop over modalities
    motion_T1s = []
    im = reg.NiftiImageData(data_path + '/T1_mf0.nii')
    template_mr_im.fill(im)
    # Loop over motion states
    for ind in range(num_ms):
        # Get TM for given motion state
        tm = reg.AffineTransformation(transform_matrices_files[ind])
        # Get resampler
        res = get_resampler_from_tm(tm, template_mr_im)
        # Resample
        motion_im = res.forward(template_mr_im)
        # Save to file as nii
        save_as_nifti(motion_im, data_path + '/T1_ms' + str(ind))
        # Store modality as necessary
        motion_T1s.append(motion_im)

    ############################################################################################
    # Create coil sensitivity data
    ############################################################################################

    csm = mr.CoilSensitivityData()
    csm.smoothness = 500
    csm.calculate(prep_data)

    ############################################################################################
    # Now create k-space data for motion states
    ############################################################################################

    # Create interleaved sampling
    mvec = []
    for ind in range(num_ms):
        mvec.append(np.arange(ind, acq.number(), num_ms))

    # Go through motion states and create k-space
    acq_ms = [0]*num_ms
    acq_ms_sim = [0]*num_ms
    for ind in range(num_ms):

        acq_ms[ind] = acq.new_acquisition_data(empty=True)

        # Set first two (??) acquisition
        acq_ms[ind].append_acquisition(acq.acquisition(0))
        acq_ms[ind].append_acquisition(acq.acquisition(1))

        # Add motion resolved data
        for jnd in range(len(mvec[ind])):
            if mvec[ind][jnd] < acq.number() - 1 and mvec[ind][jnd] > 1:  # Ensure first and last are not added twice
                cacq = acq.acquisition(mvec[ind][jnd])
                acq_ms[ind].append_acquisition(cacq)

        # Set last acquisition
        acq_ms[ind].append_acquisition(acq.acquisition(acq.number() - 1))

        # Create acquisition model
        AcqMod = mr.AcquisitionModel(acq_ms[ind], motion_T1s[ind])
        AcqMod.set_coil_sensitivity_maps(csm)

        # Forward project!
        acq_ms_sim[ind] = AcqMod.forward(motion_T1s[ind])

    for ind in range(num_ms):
        acq_ms_sim[ind] = mr.AcquisitionData("/home/rich/Documents/Data/brainweb_single_slice_256/raw_T1_ms" + str(ind) + ".h5")
        acq_ms[ind] = mr.AcquisitionData("/home/rich/Documents/Data/brainweb_single_slice_256/raw_T1_ms" + str(ind) + ".h5")



        
    ############################################################################################
    # Set up reconstructor
    ############################################################################################

    print("Setting up reconstructor...")

    AcqModMs = [0]*num_ms
    resamplers = [0]*num_ms
    prep_datas = [0]*num_ms
    for ind in range(len(acq_ms)):
        # Acquisition model for the individual motion states
        AcqModMs[ind] = mr.AcquisitionModel(acq_ms[ind], template_mr_im)
        AcqModMs[ind].set_coil_sensitivity_maps(csm)
        # Create resamplers from the transformation matrix
        tm = reg.AffineTransformation(transform_matrices_files[ind])
        resamplers[ind] = get_resampler_from_tm(tm, template_mr_im)
        # Pre-process data
        prep_datas[ind] = mr.preprocess_acquisition_data(acq_ms_sim[ind])

    acq_mods = AcqModMs
    acqs = acq_ms_sim


    # Create composition operators containing acquisition models and resamplers
    C = [CompositionOperator(am, res) for am, res in zip(*(acq_mods, resamplers))]

    print("here1")
    # Configure the PDHG algorithm
    ls = [L2NormSquared(b=data) for data in acqs]
    print("here2")
    f = BlockFunction(*ls)
    print("here3")
    K = BlockOperator(*C)
    print("here4")
    normK = K.norm(iterations=10)
    print("here5")
    #normK = LinearOperator.PowerMethod(K, iterations=10)[0]
    sigma = 0.001
    print("here6")
    tau = 1/(sigma*normK**2)
    print("here7")
    # default values
    # sigma = 1 / normK
    # tau = 1 / normK
    print("Norm of the BlockOperator ", normK)

    if regularisation == 'none':
        G = ZeroFunction()
    elif regularisation == 'FGP_TV':
        r_alpha = 5e-1
        r_iterations = 100
        r_tolerance = 1e-7
        r_iso = 0
        r_nonneg = 1
        r_printing = 0
        G = FGP_TV(r_alpha, r_iterations, r_tolerance, r_iso, r_nonneg, r_printing, 'gpu')

    pdhg = PDHG(f=f, g=G, operator=K, sigma=sigma, tau=tau,
                max_iteration=1000,
                update_objective_interval=1)

    ############################################################################################
    # Reconstruct
    ############################################################################################

    print("Reconstructing...")

    pdhg.run(num_iters, verbose=True)

    mcir_recon = pdhg.get_output()
    reg.NiftiImageData(mcir_recon).write(output_fname)


# if anything goes wrong, an exception will be thrown 
# (cf. Error Handling section in the spec)
try:
    main()
    print('done')
except error as err:
    # display error information
    print('%s' % err.value)
