# -*- coding: utf-8 -*-
"""pSTIR OSSPS reconstruction tests
v{version}

Usage:
  tests_three [--help | options]

Options:
  -r, --record   record the measurements rather than check them
  -v, --verbose  report each test status

{author}

{licence}
"""
from pSTIR import *
__version__ = "0.2.2"
__author__ = "Casper da Costa-Luis"


def test_main(rec=False, verb=False, throw=True):
    datafile = RE_PYEXT.sub(".txt", __file__)
    test = pTest(datafile, rec, throw=throw)
    test.verbose = verb

    msg_red = MessageRedirector(warn=None)

    data_path = petmr_data_path('pet')
    raw_data_file = existing_filepath(data_path, 'Utahscat600k_ca_seg4.hs')
    acq_data = AcquisitionData(raw_data_file)
    test.check(acq_data.norm())

    init_image_file = existing_filepath(data_path, 'test_image_PM_QP_6.hv')
    image_data = ImageData(init_image_file)
    test.check(image_data.norm())

    acq_model = AcquisitionModelUsingRayTracingMatrix()

    obj_fun = make_Poisson_loglikelihood(acq_data)
    obj_fun.set_acquisition_model(acq_model)
    obj_fun.set_prior(QuadraticPrior().set_penalisation_factor(0.5))

    recon = OSSPSReconstructor()
    recon.set_num_subsets(4)
    recon.set_num_subiterations(2)
    recon.set_objective_function(obj_fun)
    recon.set_input(acq_data)
    if verb:
        print('setting up, please wait...')
    recon.set_up(image_data)
    recon.set_current_estimate(image_data)
    if verb:
        print('reconstructing, please wait...')
    recon.process()
    image_data = recon.get_output()
    test.check(image_data.norm())

    return test.failed, test.ntest


if __name__ == "__main__":
    runner(test_main, __doc__, __version__, __author__)
