"""Generate a NEMA-esque phantom

Usage:
  generate_nema-esque_phantom [--help | options]

Options:
  -s <file>, --sino=<file>      template sinogam
                                (default: mMR/mMR_template_span11_small.hs)
  -i <path>, --img=<path>       template image (default: auto-generated
                                by sinogram)
  -S <path>, --out_sino=<path>  output sinogram prefix [default: sino]
  -I <path>, --out_im=<path>    output image filename [default: im]
  --min_radius=<int>            num of voxels for smallest sphere [default: 1]
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

from sirf.Utilities import examples_data_path, error
import sirf.STIR as pet
import sirf.Reg as reg
import numpy as np
from docopt import docopt
from math import pi, cos, sin

__version__ = '0.1.0'
args = docopt(__doc__, version=__version__)

# Get filenames
if args['--sino']:
    f_sino = args['--sino']
else:
    data_path = examples_data_path('PET')
    f_sino = data_path + '/mMR/mMR_template_span11.hs'
f_image = args['--img'] if args['--img'] else None
f_out_im = args['--out_im']
f_out_sino = args['--out_sino']

# Get radius of smallest sphere
min_radius = int(args['--min_radius'])


def add_sphere(image, centre, radius, intensity):
    """Add a sphere into the image"""
    for z in range(centre[0] - radius, centre[0] + radius + 1):
        for y in range(centre[1] - radius, centre[1] + radius + 1):
            for x in range(centre[2] - radius, centre[2] + radius + 1):
                dist = pow((centre[0]-z)**2 + (centre[1]-y)**2 +
                           (centre[2]-x)**2, 0.5)
                if dist <= radius:
                    image[z, y, x] = intensity


def get_acquisition_model(templ_sino, templ_im):
    """Get acquisition model"""
    am = pet.AcquisitionModelUsingRayTracingMatrix()
    am.set_num_tangential_LORs(5)
    am.set_up(templ_sino, templ_im)
    return am


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
    sino_templ = pet.AcquisitionData(f_sino)
    if f_image:
        image = pet.ImageData(f_image)
        image.fill(0.0)
    else:
        image = sino_templ.create_uniform_image(0.0)
    im_arr = image.as_array()

    # Check image dimensions
    dims = np.array(image.dimensions())
    min_dims = np.array((127, 285, 285))
    if any(dims < min_dims):
        raise pet.error(f'Image too small. Minimum size: {min_dims},
                        actual size: {dims}')
    # Get centre of image
    centre = dims//2
    # Radius of source centres is quarter of x or y dim (whichever is smaller)
    r = np.min(dims[1:2]//4)

    # Loop over number of sources
    num_sources = 5
    intensity = 10.
    for i in range(num_sources):
        # get angle. use minus pi/2 to start at top (y=0)
        theta = i * 2*pi/num_sources - pi/2
        # Point is (r,theta)->(x,y) + centre
        point = np.array((0, r*sin(theta), r*cos(theta)), dtype=np.int32)
        point += centre
        add_sphere(im_arr, point, (i+1)*min_radius, intensity)

    # Fill back into image
    image.fill(im_arr)
    # Save
    reg.ImageData(image).write(f_out_im)

    # Forward project
    am = get_acquisition_model(sino_templ, image)
    sino = am.forward(image)
    sino.write(f_out_sino + "_100-percent")

    # Different noise realisations
    count_fractions = [0.01, 0.1, 1, 10, 50]
    for i in count_fractions:
        noisy_sino = add_sino_noise(i/100, sino)
        noisy_sino.write(f_out_sino + "_" +
                         str(i).replace('.', '_') + "-percent")


# if anything goes wrong, an exception will be thrown
# (cf. Error Handling section in the spec)
try:
    main()
    print('done')
except error as err:
    # display error information
    print('%s' % err.value)