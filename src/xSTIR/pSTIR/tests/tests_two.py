'''pSTIR test set 2.

OSEM rreconstruction tests.

CCP PETMR Synergistic Image Reconstruction Framework (SIRF)
Copyright 2015 - 2017 Rutherford Appleton Laboratory STFC
2017 Casper da Costa-Luis

This is software developed for the Collaborative Computational
Project in Positron Emission Tomography and Magnetic Resonance imaging
(http://www.ccppetmr.ac.uk/).

Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
'''
from pSTIR import *

__version__ = '0.2.0'


def test_main(rec=False, verb=False):
    test = pTest('test2.txt', rec)
    test.verbose = verb

    msg_red = MessageRedirector()

    data_path = petmr_data_path('pet')
    raw_data_file = existing_filepath(data_path, 'my_forward_projection.hs')
    acq_data = AcquisitionData(raw_data_file)
    test.check(acq_data.norm())

    image = acq_data.create_uniform_image(1.0)
    test.check(image.norm())

    acq_model = AcquisitionModelUsingRayTracingMatrix()
    acq_model.set_up(acq_data, image)

    obj_fun = make_Poisson_loglikelihood(acq_data)
    obj_fun.set_acquisition_model(acq_model)

    num_subsets = 12
    recon = OSMAPOSLReconstructor()
    recon.set_objective_function(obj_fun)
    recon.set_num_subsets(num_subsets)
    recon.set_input(acq_data)
    print('setting up, please wait...')
    recon.set_up(image)

    recon.set_current_estimate(image)

    num_iterations = 2
    for iteration in range(num_iterations):
        print('\n------------- iteration %d' % iteration)
        recon.update_current_estimate()
    test.check(image.norm())

    print('projecting...')
    simulated_data = acq_model.forward(image)
    diff = simulated_data * (acq_data.norm()/simulated_data.norm()) - acq_data
    print('relative residual norm: %e' % (diff.norm()/acq_data.norm()))
    test.check(diff.norm())
