import sirf.Gadgetron as pMR
import sirf.STIR as pet
import sirf.Reg as reg
from sirf.Utilities import examples_data_path
from ccpi.utilities.display import plotter2D
from ccpi.optimisation.algorithms import PDHG
from ccpi.optimisation.functions import L2NormSquared, BlockFunction, IndicatorBox, ZeroFunction
from ccpi.optimisation.operators import CompositionOperator, BlockOperator, LinearOperator
from ccpi.plugins.regularisers import FGP_TV
import os
import numpy as np
import matplotlib.pyplot as plt
import glob

pet.AcquisitionData.set_storage_scheme('memory')

def get_resampler_from_tm(tm, image):
    """returns a NiftyResample object for the specified transform matrix and image"""
    resampler = reg.NiftyResample()
    resampler.set_reference_image(image)
    resampler.set_floating_image(image)
    resampler.add_transformation(tm)
    resampler.set_padding_value(0)
    resampler.set_interpolation_type_to_linear()
    
    return resampler


def simple_mr_recon(input_data):
    """Simple MR recon from input data"""
    recon = pMR.CartesianGRAPPAReconstructor()
    recon.set_input(input_data)
    recon.compute_gfactors(False)
    recon.process()
    return recon.get_output()


def save_as_nifti(im, filename):
    """Save an image (e.g., MR) as nii"""
    reg.NiftiImageData(im).write(filename)


# Get files
data_path = os.path.join( '/home/rich/Documents/Data', 'brainweb_single_slice_256')
transform_matrices_files  = sorted( glob.glob(os.path.join( data_path, 'fwd_tm*_trans.txt') ) )
T1_template = os.path.join( data_path, 'T1_mf0.nii')
mumap_template = os.path.join( data_path, 'mumap_mf0.nii')
FDG_template = os.path.join( data_path, 'FDG_mf0.nii')


############################################################################################
#  First reconstruct example data to use its metadata, reorient into PET space
############################################################################################

template_acq_data = acq = pMR.AcquisitionData(os.path.join(data_path , 'CSM_FULLY_FOV534.h5' ))
template_acq_data.sort_by_time()
prep_data = pMR.preprocess_acquisition_data(template_acq_data)

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
for modality in ['T1','uMap','FDG']:
    im = reg.NiftiImageData(data_path + '/' + modality + '_mf0.nii')
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
        save_as_nifti(motion_im, data_path + '/' + modality + '_ms' + str(ind))
        # Store modality as necessary
        if modality == 'T1':
            motion_T1s.append(motion_im)

############################################################################################
# Create coil sensitivity data
############################################################################################

csm = pMR.CoilSensitivityData()
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
    AcqMod = pMR.AcquisitionModel(acq_ms[ind], motion_T1s[ind])
    AcqMod.set_coil_sensitivity_maps(csm)

    # Forward project!
    acq_ms_sim[ind] = AcqMod.forward(motion_T1s[ind])

    # Save
    print("writing: " + data_path + '/raw_T1_ms' + str(ind) + '.h5')
    acq_ms_sim[ind].write(data_path + '/raw_T1_ms' + str(ind) + '.h5')

    # Back project and save each individually to check orientation
    back = AcqMod.backward(acq_ms_sim[ind])
    save_as_nifti(back, data_path + '/back_T1_ms' + str(ind))


############################################################################################
# Perform MCIR
############################################################################################

AcqModMs = [0]*num_ms
resamplers = [0]*num_ms
prep_datas = [0]*num_ms

for ind in range(len(acq_ms)):
    # Acquisition model for the individual motion states
    AcqModMs[ind] = pMR.AcquisitionModel(acq_ms[ind], template_mr_im)
    AcqModMs[ind].set_coil_sensitivity_maps(csm)
    # Create resamplers from the transformation matrix
    tm = reg.AffineTransformation(transform_matrices_files[ind])
    resamplers[ind] = get_resampler_from_tm(tm, template_mr_im)
    # Pre-process data
    prep_datas[ind] = pMR.preprocess_acquisition_data(acq_ms_sim[ind])

# Create composition operators containing acquisition models and resamplers
C = [ CompositionOperator(am, res) for am, res in zip (*(AcqModMs, resamplers)) ]

# Configure the PDHG algorithm
# if simulated data, use acq_data_sim. For real data, use prep_datas
ls = [ L2NormSquared(b=data) for data in acq_data_sim ]
# ls = [ L2NormSquared(b=data) for data in prep_datas ]
f = BlockFunction(*ls)
K = BlockOperator(*C)
normK = K.norm(iterations=10)
#normK = LinearOperator.PowerMethod(K, iterations=10)[0]
# default values
sigma = 1/normK
tau = 1/normK 
sigma = 0.001
tau = 1/(sigma*normK**2)
print ("Norm of the BlockOperator ", normK)

# TV regularisation
#regularisation parameters for TV
r_alpha = 5e-1
r_iterations = 100
r_tolerance = 1e-7
r_iso = 0
r_nonneg = 1
r_printing = 0


# TV = FGP_TV(r_alpha, r_iterations, r_tolerance, r_iso,r_nonneg,r_printing,'gpu')

print ("current dir", os.getcwd())

# G = IndicatorBox(lower=0)
# G = TV
G = ZeroFunction()
pdhg = PDHG(f = f, g = G, operator = K, sigma = sigma, tau = tau, 
            max_iteration = 1000,
            update_objective_interval = 1)

pdhg.run(2, verbose=True)

MCIR_recon = pdhg.get_output()
save_as_nifti(MCIR_recon, data_path + '/MCIR_recon_iter_2')

pdhg.run(8, verbose=True)

MCIR_recon = pdhg.get_output()
save_as_nifti(MCIR_recon, data_path + '/MCIR_recon_iter_10')
