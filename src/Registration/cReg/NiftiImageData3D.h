/*
CCP PETMR Synergistic Image Reconstruction Framework (SIRF)
Copyright 2015 - 2017 Rutherford Appleton Laboratory STFC

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
\brief Base class for SIRF image data.

\author Richard Brown
\author CCP PETMR
*/

#ifndef _NIFTIIMAGEDATA3D_H_
#define _NIFTIIMAGEDATA3D_H_

#include "NiftiImageData.h"
#include <nifti1_io.h>
#include <string>
#include <memory>
#include <iostream>

template <int num_dimensions>
class VoxelisedGeometricalInfo;
/// Typedef VoxelisedGeometricalInfo for 3D
typedef VoxelisedGeometricalInfo<3> VoxelisedGeometricalInfo3D;

namespace sirf {

class PETImageData;
class MRImageData;

/// SIRF image data
class NiftiImageData3D : public NiftiImageData
{
public:

    /// Constructor
    NiftiImageData3D() {}

    /// Construct 3D from general case
    NiftiImageData3D(const NiftiImageData& general)
        : NiftiImageData(general) { check_dimensions(_3D); }

    /// Filename constructor
    NiftiImageData3D(const std::string &filename)
        : NiftiImageData(filename)
    { check_dimensions(_3D); }

    /// Nifti constructor
    NiftiImageData3D(const nifti_image &image_nifti)
        : NiftiImageData(image_nifti) { check_dimensions(_3D); }

    /// Nifti std::shared_ptr constructor
    NiftiImageData3D(const std::shared_ptr<nifti_image> image_nifti)
        : NiftiImageData(image_nifti) { check_dimensions(_3D); }

    /// STIR constructor
    NiftiImageData3D(const PETImageData &pet_image);

    /// Gadgetron constructor
    NiftiImageData3D(const MRImageData &);

    /// Copy data to PETImageData
    void copy_data_to(PETImageData &pet_image) const;

    /// Copy data to MRImageData
    void copy_data_to(MRImageData &) const;

    /// Deep copy
    NiftiImageData3D deep_copy() const
    { return this->NiftiImageData::deep_copy(); }

protected:

    /// Set up nifti image
    void set_up_nifti(const VoxelisedGeometricalInfo3D &info);

    /// Check that images are aligned
    bool check_images_are_aligned(const VoxelisedGeometricalInfo3D &info) const;
};
}

#endif
