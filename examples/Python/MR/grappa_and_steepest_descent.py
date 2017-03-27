'''
GRAPPA reconstruction with the steepest descent step: illustrates
the use of Acquisition Model projections

Usage:
  grappa_and_steepest_descent.py [--help | options]

Options:
  -f <file>, --file=<file>    raw data file
                              [default: simulated_MR_2D_cartesian_Grappa2.h5]
  -p <path>, --path=<path>    path to data files, defaults to data/examples/MR
                              subfolder of SIRF root folder
  -e <engn>, --engine=<engn>  reconstruction engine [default: Gadgetron]
'''

## CCP PETMR Synergistic Image Reconstruction Framework (SIRF)
## Copyright 2015 - 2017 Rutherford Appleton Laboratory STFC
## Copyright 2015 - 2017 University College London.
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

__version__ = '0.1.0'
from docopt import docopt
args = docopt(__doc__, version=__version__)

try:
    import pylab
    HAVE_PYLAB = True
except:
    HAVE_PYLAB = False

# import engine module
exec('from p' + args['--engine'] + ' import *')

def main():

    # locate the input data file
    data_path = args['--path']
    if data_path is None:
        data_path = mr_data_path()
    input_file = existing_filepath(data_path, args['--file'])

    # acquisitions will be read from an HDF file input_data
    input_data = AcquisitionData(input_file)

    # pre-process acquisitions
    print('---\n pre-processing acquisition data...')
    preprocessed_data = preprocess_acquisition_data(input_data)

    # perform reconstruction
    recon = CartesianGRAPPAReconstruction()
    recon.set_input(preprocessed_data)
    recon.compute_gfactors(False)
    print('---\n reconstructing...')
    recon.process()
    # for undersampled acquisition data GRAPPA computes Gfactor images
    # in addition to reconstructed ones
    complex_images = recon.get_output()

    # compute coil sensitivity maps
    csms = CoilSensitivityData()
    print('---\n sorting acquisitions...')
    preprocessed_data.sort()
    print('---\n computing sensitivity maps...')
    csms.calculate(preprocessed_data)

    # create acquisition model based on the acquisition parameters
    # stored in preprocessed_data and image parameters stored in complex_images
    am = AcquisitionModel(preprocessed_data, complex_images)
    am.set_coil_sensitivity_maps(csms)

    # use the acquisition model (forward projection) to simulate acquisitions
    fwd_data = am.forward(complex_images)

    # compute the difference between real and simulated acquisitions
    pp_norm = preprocessed_data.norm()
    fwd_norm = fwd_data.norm()
    res = fwd_data - preprocessed_data * (fwd_norm/pp_norm)
    rr = res.norm()/fwd_norm
    print('---\n reconstruction residual norm (rel): %e' % rr)

    # try to improve the reconstruction by the steepest descent step
    grad = am.backward(res)
    w = am.forward(grad)
    alpha = (grad*grad)/(w*w)
    refined_cmplx_imgs = complex_images - grad*alpha # refined images

    image_as_3D_array = abs(complex_images.as_array())
    refined_image_as_3D_array = abs(refined_cmplx_imgs.as_array())
    nz = image_as_3D_array.shape[0]

    # show images
    while HAVE_PYLAB:
        print('---\n Enter the slice number to view it.')
        print(' A value outside the range [1 : %d] will stop this loop.'% nz)
        s = str(input('---\n slice: '))
        if len(s) < 1:
            break
        z = int(s)
        if z < 1 or z > nz:
            break
        pylab.figure(z)
        pylab.title('image')
        pylab.imshow(image_as_3D_array[z - 1, :, :])
        print(' Close Figure %d window to continue...' % z)
        pylab.figure(z + nz)
        pylab.title('refined image')
        pylab.imshow(refined_image_as_3D_array[z - 1, :, :])
        print(' Close Figure %d window to continue...' % (z + nz))
        pylab.show()

try:
    main()
    print('done')

except error as err:
    # display error information
    print('??? %s' % err.value)