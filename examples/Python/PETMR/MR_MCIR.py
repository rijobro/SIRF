import sirf.Gadgetron as pMR
import sirf.STIR as pet
import sirf.Reg as reg
from sirf.Utilities import examples_data_path
from ccpi.utilities.display import plotter2D
#from PETMR_MCIR import get_resampler_from_tm

from ccpi.optimisation.algorithms import PDHG
from ccpi.optimisation.functions import LeastSquares, BlockFunction, IndicatorBox, ZeroFunction
from ccpi.optimisation.operators import CompositionOperator, BlockOperator, LinearOperator
from ccpi.plugins.regularisers import FGP_TV

def get_resampler_from_tm(tm, image):
    '''returns a NiftyResample object for the specified transform matrix and image'''

    mat = tm.as_array()

    resampler = reg.NiftyResample()
    resampler.set_reference_image(image)
    resampler.set_floating_image(image)
    resampler.add_transformation(tm)
    resampler.set_padding_value(0)
    resampler.set_interpolation_type_to_linear()
    
    return resampler
pet.AcquisitionData.set_storage_scheme('memory')

#%% Go to directory with input files
# Adapt this path to your situation (or start everything in the relevant directory)
#os.chdir(examples_data_path('PET'))

# import further modules
import os
import numpy as np
# import hdf5storage

import matplotlib.pyplot as plt

import glob

plt.close('all')

# path- and filename of raw data file
#pname = '/media/sf_Data/MRI/Output/PSMR_abstract/'
pname = os.path.abspath(examples_data_path('MR'))
fname = 'grappa2_1rep.h5'


data_path = os.path.join( os.path.abspath('/home/ofn77899') ,
                          'brainweb', 'brainweb_single_slice_256')

T1_files   = sorted( glob.glob(os.path.join( data_path, 'T1_mf*.nii') ) )
T2_files   = sorted( glob.glob(os.path.join( data_path, 'T2_mf*.nii') ) )
transform_matrices_files  = sorted( glob.glob(os.path.join( data_path, 'fwd_tm*.txt') ) )
for f in T1_files:
    print (f)


# chdir to data dir
# os.chdir(data_path)
# Number of motion states
num_ms = 4

## Create motion states

# Split k-space data into different motion states
# acq = pMR.AcquisitionData(os.path.join(data_path , 'CSM_GRAPPA2_48_FOV180.h5' ))

acq = pMR.AcquisitionData(os.path.join(data_path , 'CSM_FULLY_FOV180.h5' ))

# acq = pMR.AcquisitionData(os.path.join(os.path.abspath('/home/ofn77899') ,
#                           'brainweb' , 'CSM_GRAPPA2_48_FOV180.h5' ) )

# acq = pMR.AcquisitionData(os.path.join(pname , fname ))
acq.sort_by_time()

# Create interleaved sampling
mvec = []
for ind in range(num_ms):
    mvec.append(np.arange(ind, acq.number(), num_ms))

# Go through motion states and create k-space
acq_ms = [0]*num_ms
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

    #acq_ms[ind].write(pname + fname +  '_ms' + str(ind) + '.h5')
    print('MS {} with {} points'.format(ind, acq_ms[ind].number()))


## Calculate reference image and coil maps
prep_data = pMR.preprocess_acquisition_data(acq)

recon = pMR.CartesianGRAPPAReconstructor()
recon.set_input(prep_data)
recon.compute_gfactors(False)
recon.process()
ref_im = recon.get_output()

ref_im.get_geometrical_info().print_info()


print ("ref_im shape", ref_im.shape)

fig, ax = plt.subplots(1)
ax.imshow(np.abs(ref_im.as_array()[0,:,:]))

csm = pMR.CoilSensitivityData()
csm.smoothness = 500
csm.calculate(prep_data)

fig, ax = plt.subplots(1,4)
for ind in range(4):
    ax[ind].imshow(np.abs(csm.as_array()[ind,0,:,:]))
# plt.show()

t1s = []
for t1f in T1_files:
    t1s.append(
        reg.NiftiImageData(t1f).as_array()
    )
    print (t1s[-1].shape)
# plotter2D(t1s)

## Simulate different motion states
ref_im_ms = ref_im.clone()
print("ref_im")
ref_im.get_geometrical_info().print_info()
print("ref_im_ms")
ref_im_ms.get_geometrical_info().print_info()
acq_ms_sim = [0]*num_ms
for ind in range(num_ms):
    AcqMod = pMR.AcquisitionModel(acq_ms[ind], ref_im)
    AcqMod.set_coil_sensitivity_maps(csm)
    ## @EDO: Here you will need your rotation transformation and the BrainWeb image

    if False:
        # The saved brainweb images are real, hence requires casting to complex
        t1 = np.asarray( reg.NiftiImageData(T1_files[ind]).as_array(), dtype=np.complex64)
        # The AcquisitionModel range has shape 1,256,256
        # We plan to put in the brainweb data that are 150,150, so requires this requires padding
        cim = np.pad(t1, ((53,53), (53,53)), 'constant', constant_values=0)
        # and finally reshape to (1,256,256)
        cim = np.reshape(cim, (1,256,256))
    else:
        cim = ref_im.as_array()
        # cim = np.roll(cim, ind*5, axis=1) 
        cim = np.asarray( reg.NiftiImageData(T1_files[ind]).as_array(), dtype=np.complex64)
    print ("type", cim.dtype, cim.shape)
    ref_im_ms.fill(cim)
    print("ref_im_ms")
    ref_im_ms.get_geometrical_info().print_info()


    acq_ms_sim[ind] = AcqMod.forward(ref_im_ms)


## Reconstruct motion states
fig, ax = plt.subplots(1,num_ms+1)

im_sum = 0
AcqModMs = [0]*num_ms
resamplers = []
for ind in range(len(acq_ms)):
    prep_data = pMR.preprocess_acquisition_data(acq_ms_sim[ind])

    ## @EDO: With these two lines you define the acquisition model for the individual motion states
    AcqModMs[ind] = pMR.AcquisitionModel(acq_ms[ind], ref_im)
    AcqModMs[ind].set_coil_sensitivity_maps(csm)
    # create the resamplers from the TransformationMatrix
    tm = reg.AffineTransformation(transform_matrices_files[ind])
    resamplers.append(

        get_resampler_from_tm(tm, ref_im)
 
    )
if True:
    # plot the solutions to see that with the resampler we get the same orientation
    implot = []
    g_refim = ref_im.copy()
    for ms in range(len(acq_ms)):
        img = reg.NiftiImageData(T1_files[ms])
        implot.append(img.as_array())
        if False:
            a = reg.NiftiImageData(T1_files[ms])
            tm = reg.AffineTransformation(transform_matrices_files[ms])
            r = get_resampler_from_tm(tm, a)
            b = r.adjoint(a)
            t1 = np.asarray( b.as_array(), dtype=np.complex64)
        else:
            # The saved brainweb images are real, hence requires casting to complex
            cim = np.asarray( reg.NiftiImageData(T1_files[ms]).as_array(), dtype=np.complex64)
        # # The AcquisitionModel range has shape 1,256,256
        # # We plan to put in the brainweb data that are 150,150, so requires this requires padding
        # cim = np.pad(t1, ((0,106), (0,106)), 'constant', constant_values=0)
        # # and finally reshape to (1,256,256)
        # cim = np.reshape(cim, (1,256,256))
        g_refim.fill(cim)
        #print ("adjoint ", resamplers[ms].adjoint(g_refim).as_array().shape)
        implot.append(np.abs(
            resamplers[ms].adjoint(g_refim).as_array()[0])
        )

    plotter2D( implot , titles=['MS0', 'Resampled to MS0', 
    'MS1', 'Resampled to MS0',
    'MS2', 'Resampled to MS0',
    'MS3', 'Resampled to MS0'])

# compose the resampler with the acquisition models
C = [ CompositionOperator(am, res) for am, res in zip (*(AcqModMs, resamplers)) ]
# C = [ am for am in ams ]
# C = [ CompositionOperator(am, resamplers[0]) for am in ams ]
print ("number of motion states", len(resamplers))


# Configure the PDHG algorithm

# kl = [ KullbackLeibler(b=rotated_sino, eta=(rotated_sino * 0 + 1e-5)) for rotated_sino in rotated_sinos ] 
ls = [ LeastSquares(A=am , b=data) for am, data in zip(* (AcqModMs, acq_ms_sim)) ]
f = BlockFunction(*ls)
K = BlockOperator(*C)

#f = kl[0]
#K = ams[0]

for i in range(len(C)):
    print ("##################################### " , i)
    a = C[i].direct(ref_im)
    if True:
        c = AcqModMs[i].adjoint(a)
        b = resamplers[i].adjoint(c)
    else:
        c.write('blah')
        d = pMR.ImageData('blah.h5')
        b = resamplers[i].adjoint(d)
    
    
    b = C[i].adjoint(a)

x0 = ref_im.copy()
x1 = ref_im.copy()
for i in range(2):
    print ("POWER METHOD ##################################### " , i)
    
    x0.get_geometrical_info().print_info()
    a = K.direct(x0)
    K.adjoint(a, out=x1)
    x1norm = x1.norm()

    print ("before multiply ##################################### " , i)
    
    x0.get_geometrical_info().print_info()
    x1.multiply((1.0/x1norm), out=x0)
    



def PowerMethod(operator, iterations, x_init=None):
    '''Power method to calculate iteratively the Lipschitz constant
    
    :param operator: input operator
    :type operator: :code:`LinearOperator`
    :param iterations: number of iterations to run
    :type iteration: int
    :param x_init: starting point for the iteration in the operator domain
    :returns: tuple with: L, list of L at each iteration, the data the iteration worked on.
    '''
    
    # Initialise random
    if x_init is None:
        x0 = operator.domain_geometry().allocate('random')
    else:
        x0 = x_init.copy()
        
    x1 = operator.domain_geometry().allocate()
    y_tmp = operator.range_geometry().allocate()
    s = np.zeros(iterations)
    # Loop
    for it in np.arange(iterations):
        operator.direct(x0, out=y_tmp)
        operator.adjoint(y_tmp, out=x1)
        x1norm = x1.norm()
        if hasattr(x0, 'squared_norm'):
            s[it] = x1.dot(x0) / x0.squared_norm()
        else:
            x0norm = x0.norm()
            s[it] = x1.dot(x0) / (x0norm * x0norm) 
        x1.multiply((1.0/x1norm), out=x0)
    return np.sqrt(s[-1]), np.sqrt(s), x0




# normK = PowerMethod(K, 25, ref_im)
normK = K.norm(iterations=10)
#normK = LinearOperator.PowerMethod(K, iterations=10)[0]
#default values
sigma = 1/normK
tau = 1/normK 
sigma = 0.001
tau = 1/(sigma*normK**2)
print ("Norm of the BlockOperator ", normK)

    
# TV regularisation
#regularisation parameters for TV
# 
r_alpha = 5e-2
r_iterations = 100
r_tolerance = 1e-7
r_iso = 0
r_nonneg = 1
r_printing = 0

TV = FGP_TV(r_alpha, r_iterations, r_tolerance, r_iso,r_nonneg,r_printing,'gpu')

print ("current dir", os.getcwd())

# G = IndicatorBox(lower=0)
# G = TV
G = ZeroFunction()
pdhg = PDHG(f = f, g = G, operator = K, sigma = sigma, tau = tau, 
            max_iteration = 1000,
            update_objective_interval = 1)

pdhg.run(100, verbose=False)
#img = convert_nifti_to_stir_ImageData(reg.NiftiImageData(os.path.basename(FDG_files[0])), templ_sino, dim)
# img = read_2D_STIR_nii(os.path.basename(FDG_files[0]))
solution = reg.NiftiImageData(T1_files[0]).as_array()


# res = reg.NiftyResample()
# res.set_reference_image(pdhg.get_output())
# res.set_floating_image(solution)
# res.set_interpolation_type_to_linear()

# solution2 = res.direct(solution)

plotter2D([solution.as_array(), pdhg.get_output().as_array()[0]],
          titles = ['Ground Truth (MS0)' , 'PDHG output with TV'] )