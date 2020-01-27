/*
CCP PETMR Synergistic Image Reconstruction Framework (SIRF)
Copyright 2017 - 2019 University College London

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

*/

/*!
\file
\ingroup Registration
\brief NiftyReg's aladin class for rigid and affine registrations.

\author Richard Brown
\author CCP PETMR
*/

#include "sirf/Reg/SPM12Registration.h"
#include "sirf/Reg/NiftiImageData3D.h"
#include "sirf/Reg/NiftiImageData.h"
#include <sys/stat.h>
#include <MatlabEngine.hpp>
#include <boost/filesystem.hpp>

using namespace sirf;
using namespace matlab::engine;
using namespace matlab::data;

inline bool check_file_exists(const std::string& filename, const bool existance_allowed) {
    struct stat buffer;
    const bool file_exists (stat (filename.c_str(), &buffer) == 0);
    if (file_exists && !existance_allowed)
        throw std::runtime_error("SPM12Registration<dataType>::process(): file exists: " + filename);
    return file_exists;
}

template<class dataType>
std::shared_ptr<const StructArray> convert_to_spmvol_from_NiftiImageData3D(std::shared_ptr<const NiftiImageData3D<dataType> > in_sptr)
{
    const nifti_image * const in_ptr = in_sptr->get_raw_nifti_sptr().get();
    ArrayFactory factory;

    // Convert voxel data to matlab array
    TypedArray<double> dat =
            factory.createArray<double>({unsigned(in_ptr->nx), unsigned(in_ptr->ny), unsigned(in_ptr->nz)});
    int idx[7] = {0,0,0,0,0,0,0};
    for (idx[0]=0; idx[0]<in_ptr->nx; ++idx[0])
        for (idx[1]=0; idx[1]<in_ptr->ny; ++idx[1])
            for (idx[2]=0; idx[2]<in_ptr->nz; ++idx[2])
                dat[unsigned(idx[0])][unsigned(idx[1])][unsigned(idx[2])] =
                        double((*in_sptr)(idx));

    // qto_xyz mat
    TypedArray<double> qto_xyz = factory.createArray<double>({4,4});
    for (unsigned i=0; i<4; ++i)
        for (unsigned j=0; j<4; ++j)
            qto_xyz[i][j] = double(in_ptr->qto_xyz.m[i][j]);

    // Create the private array
    StructArray priv = factory.createStructArray({1}, {
        "dat", "mat", "mat_intent", "mat0", "mat0_intent"});
    priv[0]["dat"]         = dat;
    priv[0]["mat"]         = qto_xyz;
    priv[0]["mat_intent"]  = factory.createCharArray("Aligned");
    priv[0]["mat0"]        = qto_xyz;
    priv[0]["mat0_intent"] = factory.createCharArray("Aligned");

    const std::shared_ptr<StructArray> out_sptr =
            std::make_shared<StructArray>(factory.createStructArray({1}, {
                "fname", "dim", "dt", "pinfo", "mat", "n", "descrip", "private"}));
    (*out_sptr)[0]["fname"]   = factory.createCharArray(in_ptr->fname);
    (*out_sptr)[0]["dim"]     = factory.createArray<double>({ 1,3 }, { double(in_ptr->nx), double(in_ptr->ny), double(in_ptr->nz) });
    (*out_sptr)[0]["dt"]      = factory.createArray<double>({ 1,2 }, { 4,0 }); // Type float and endian
    (*out_sptr)[0]["pinfo"]   = factory.createArray<double>({ 3,1 }, { 1,0,0 }); // slope, intercept, byte offset
    (*out_sptr)[0]["mat"]     = qto_xyz;
    (*out_sptr)[0]["n"]       = factory.createArray<double>({ 1,2 }, { 1,1 });
    (*out_sptr)[0]["descrip"] = factory.createCharArray("");
    (*out_sptr)[0]["private"] = priv;

    return out_sptr;
}

template<class dataType>
void convert_to_NiftiImageData3D_if_not_already(std::shared_ptr<const NiftiImageData3D<dataType> > &output_sptr, const std::shared_ptr<const ImageData> &input_sptr)
{
    // Try to dynamic cast from ImageData to (const) NiftiImageData. This will only succeed if original type was NiftiImageData
    output_sptr = std::dynamic_pointer_cast<const NiftiImageData3D<dataType> >(input_sptr);
    // If output is a null pointer, it means that a different image type was supplied (e.g., STIRImageData).
    // In this case, construct a NiftiImageData
    if (!output_sptr)
        output_sptr = std::make_shared<const NiftiImageData3D<dataType> >(*input_sptr);
}

template<class dataType>
void SPM12Registration<dataType>::set_working_folder(const std::string &working_folder)
{
    // Make sure it's absolute
    _working_folder = boost::filesystem::absolute(working_folder).string();
}

template<class dataType>
void SPM12Registration<dataType>::process()
{
    // Check the paramters that are NOT set via the parameter file have been set.
    this->check_parameters();

    // Filenames
    const std::string ref_filename = _working_folder + "/ref.nii";
    const std::string flo_filename = _working_folder + "/flo.nii";
    check_file_exists(ref_filename, _working_folder_overwrite);
    check_file_exists(flo_filename, _working_folder_overwrite);

    // Convert images to matlab::data::StructArray with structure expected by SPM
    std::shared_ptr<const NiftiImageData3D<dataType> > ref_nifti_sptr, flo_nifti_sptr;
    convert_to_NiftiImageData3D_if_not_already(ref_nifti_sptr, this->_reference_image_sptr);
    convert_to_NiftiImageData3D_if_not_already(flo_nifti_sptr, this->_floating_image_sptr);
    const std::shared_ptr<const StructArray> ref_spm_sptr = convert_to_spmvol_from_NiftiImageData3D(ref_nifti_sptr);
    const std::shared_ptr<const StructArray> flo_spm_sptr = convert_to_spmvol_from_NiftiImageData3D(ref_nifti_sptr);

    // Start MATLAB engine synchronously
    std::unique_ptr<MATLABEngine> matlabPtr = startMATLAB();
    std::cout << "Started MATLAB Engine" << std::endl;

    // Create MATLAB data array factory
    ArrayFactory factory;

    // Create cell array for filenames {'ref.nii','flo.nii'}
    const unsigned num_images = 2;
    CellArray spm_filenames = factory.createCellArray({num_images},
        *ref_spm_sptr,
        *flo_spm_sptr);

    // Create struct array for parameters: struct('quality',1,'rtm',1))
    StructArray spm_params = factory.createStructArray({1}, { "quality", "rtm" });
    spm_params[0]["quality"] = factory.createScalar<int>(1);
    spm_params[0]["rtm"] = factory.createScalar<int>(1);

    // Create a vector of input arguments
    std::vector<Array> args({
        spm_filenames,
        spm_params
    });

    // Call spm_realign
    const size_t num_returned = 0;
    matlabPtr->feval(u"spm_realign", num_returned, args);
}

template<class dataType>
void SPM12Registration<dataType>::check_parameters() const
{
    // Call base class
    Registration<dataType>::check_parameters();

    if (_working_folder.empty())
        throw std::runtime_error("SPM12Registration<dataType>::check_parameters(): Missing working folder.");
}

namespace sirf {
template class SPM12Registration<float>;
}
