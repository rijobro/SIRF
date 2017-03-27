'''
Medium-level demo demonstrating how 2D coil sensitivity maps can be obtained 
from a multi-coil 2D Cartesian MR acquisition

Usage:
  coil_sensitivity_maps.py [--help | options]

Options:
  -f <file>, --file=<file>    raw data file
                              [default: simulated_MR_2D_cartesian.h5]
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

import matplotlib.pyplot as plt

# import engine module
exec('from p' + args['--engine'] + ' import *')

try:
    from ismrmrdtools import coils
except:
    print('This demo requires ismrmrd-python-tools.')
    sys.exit()

def show_3D_array(array, tile_shape, scale, titles):
    assert numpy.prod(tile_shape) >= array.shape[0],\
            "tile rows x columns must equal the 3rd dim extent of array"
    cols, rows = tile_shape
    vmin, vmax = scale
    fig = plt.figure()
    for z in range(array.shape[0]):
        ax = fig.add_subplot(cols, rows, z+1)
        ax.set_title(titles[z])
        ax.set_axis_off()
        imgplot = ax.imshow(array[z,:,:], vmin=vmin, vmax=vmax)
    print('close figure 1 to continue')
    plt.show()

def main():

    # locate the input data file
    data_path = args['--path']
    if data_path is None:
        data_path = mr_data_path()
    input_file = existing_filepath(data_path, args['--file'])

    # acquisitions will be read from an HDF file input_file
    input_data = AcquisitionData(input_file)
    
    # pre-process acquisition data
    processed_data = preprocess_acquisition_data(input_data)
    
    # sort k-space data into a 2D Cartesian matrix for each coil
    processed_data.sort()
    
    # create object containing images for each coil
    CIs = CoilImageData()
    CIs.calculate(processed_data)

    # create coil sensitivity object
    CSMs = CoilSensitivityData()

    # calculate coil sensitivity maps by dividing each coil image data by the
    # Square-Root-of-the-Sum-of-Squares over all coils (SRSS)
    # (niter = 10) applies an iterative smoothing algorithm with 10 iterations 
    # to the image data prior to the calculation of the coil sensitivity maps
    CSMs.calculate(CIs, method = 'SRSS(niter = 10)')

    # display coil sensitivity maps
    coil_images = numpy.squeeze(CSMs.as_array(0))
    maxv = numpy.amax(abs(coil_images))
    show_3D_array(abs(coil_images[0::2,:,:]), tile_shape = (1,4), scale = (0, maxv),\
        titles = ['Abs(Coil1)', 'Abs(Coil3)','Abs(Coil5)','Abs(Coil7)'])
    show_3D_array(numpy.angle(coil_images[0::2,:,:]), tile_shape = (1,4), scale = (0, maxv),\
        titles = ['Angle(Coil1)', 'Angle(Coil3)','Angle(Coil5)','Angle(Coil7)']) 

    # calculate coil sensitivity maps directly from the raw k-space data 
    # so far no additional parameters can be set for this method such as the
    # number of smoothing iterations which leads to noisier coil sensitivity 
    # maps    
    CSMs = CoilSensitivityData()    
    CSMs.calculate(processed_data)
    
    # display coil sensitivity maps
    coil_images = numpy.squeeze(CSMs.as_array(0))
    maxv = numpy.amax(abs(coil_images))
    show_3D_array(abs(coil_images[0::2,:,:]), tile_shape = (1,4), scale = (0, maxv),\
        titles = ['Abs(Coil1)', 'Abs(Coil3)','Abs(Coil5)','Abs(Coil7)'])

    # calculate coil sensitivity maps using an approach suggested by 
    #   Inati SJ, Hansen MS, Kellman P.
    #   A solution to the phase problem in adaptive coil combination.
    #   In: ISMRM proceeding; April; Salt Lake City, Utah, USA; 2013. 2672.  
    # for more details please see 
    # gadgetron/toolboxes/mri_core/mri_core_coil_map_estimation.h  
    CSMs = CoilSensitivityData()
    CSMs.calculate(CIs, method = 'Inati()')
        
    # display coil sensitivity maps
    coil_images = numpy.squeeze(CSMs.as_array(0))
    maxv = numpy.amax(abs(coil_images))
    show_3D_array(abs(coil_images[0::2,:,:]), tile_shape = (1,4), scale = (0, maxv),\
        titles = ['Abs(Coil1)', 'Abs(Coil3)','Abs(Coil5)','Abs(Coil7)'])

try:
    main()
    print('done')

except error as err:
    # display error information
    print('??? %s' % err.value)