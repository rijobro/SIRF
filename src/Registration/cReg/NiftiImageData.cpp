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

#include "NiftiImageData.h"
#include "SIRFRegMisc.h"
#include <nifti1_io.h>
#include <_reg_tools.h>
#include "stir_data_containers.h"
#include "NiftiImageData3DTensor.h"
#include "NiftiImageData3DDeformation.h"
#include "NiftiImageData3DDisplacement.h"
#include "SIRFRegAffineTransformation.h"

using namespace sirf;

NiftiImageData::NiftiImageData(const NiftiImageData& to_copy)
{
    SIRFRegMisc::copy_nifti_image(_nifti_image,to_copy._nifti_image);
    set_up_data(to_copy._original_datatype);
}

NiftiImageData& NiftiImageData::operator=(const NiftiImageData& to_copy)
{
    // Check for self-assignment
    if (this != &to_copy) {
        // Check the image is copyable
        if (!to_copy.is_initialised())
            throw std::runtime_error("Trying to copy an uninitialised image.");
        // Copy
        SIRFRegMisc::copy_nifti_image(_nifti_image,to_copy._nifti_image);
        set_up_data(to_copy._original_datatype);
    }
    return *this;
}

NiftiImageData::NiftiImageData(const std::string &filename)
{
    SIRFRegMisc::open_nifti_image(_nifti_image,filename);
    set_up_data(_nifti_image->datatype);
}

NiftiImageData::NiftiImageData(const nifti_image &image_nifti)
{
    SIRFRegMisc::copy_nifti_image(_nifti_image,std::make_shared<nifti_image>(image_nifti));
    reg_checkAndCorrectDimension(_nifti_image.get());
    set_up_data(_nifti_image->datatype);
}

NiftiImageData::NiftiImageData(const std::shared_ptr<nifti_image> image_nifti)
{
    SIRFRegMisc::copy_nifti_image(_nifti_image,image_nifti);
    reg_checkAndCorrectDimension(_nifti_image.get());
    set_up_data(_nifti_image->datatype);
}

bool NiftiImageData::operator==(const NiftiImageData &other) const
{
    if (this == &other)
        return true;
    return this->are_equal_to_given_accuracy(other,1.e-3F);
}

/// Equality operator
bool NiftiImageData::operator!=(const NiftiImageData &other) const
{
    return !(*this == other);
}

NiftiImageData NiftiImageData::operator+(const NiftiImageData& c) const
{
    return maths(c, add);
}

NiftiImageData NiftiImageData::operator-(const NiftiImageData& c) const
{
    return maths(c, sub);
}

NiftiImageData NiftiImageData::operator+(const float& val) const
{
    return maths(val,add);
}

NiftiImageData NiftiImageData::operator-(const float& val) const
{
    return maths(val,sub);
}

NiftiImageData NiftiImageData::operator*(const float& val) const
{
    return maths(val,mul);
}

float NiftiImageData::operator()(const int index) const
{
    assert(this->is_in_bounds(index));
    return _data[index];
}

float &NiftiImageData::operator()(const int index)
{
    assert(this->is_in_bounds(index));
    return _data[index];
}

float NiftiImageData::operator()(const int index[7]) const
{
    assert(this->is_in_bounds(index));
    const int index_1d = this->get_1D_index(index);
    return _data[index_1d];
}

float &NiftiImageData::operator()(const int index[7])
{
    assert(this->is_in_bounds(index));
    const int index_1d = this->get_1D_index(index);
    return _data[index_1d];
}

std::shared_ptr<const nifti_image> NiftiImageData::get_raw_nifti_sptr() const
{
    if (!_nifti_image)
        throw std::runtime_error("Warning, nifti has not been initialised.");
    return _nifti_image;
}

std::shared_ptr<nifti_image> NiftiImageData::get_raw_nifti_sptr()
{
    if (!_nifti_image)
        throw std::runtime_error("Warning, nifti has not been initialised.");
    return _nifti_image;
}

void NiftiImageData::save_to_file(const std::string &filename, const int datatype) const
{
    if (!this->is_initialised())
        throw std::runtime_error("Cannot save image to file.");

    std::cout << "\nSaving image to file (" << filename << ")..." << std::flush;

    SIRFRegMisc::check_folder_exists(filename);

    if (_original_datatype == -1)
        throw std::runtime_error("Original datatype was not set.");

    // Create a deep copy in case we need to change datatype
    NiftiImageData copy = this->deep_copy();

    // If user wants to save in a different datatype
    if (datatype != -1)
        copy.change_datatype(datatype);
    else
        copy.change_datatype(_original_datatype);

    nifti_set_filenames(copy._nifti_image.get(), filename.c_str(), 0, 0);
    nifti_image_write(copy._nifti_image.get());
    std::cout << "done.\n\n";
}

float NiftiImageData::get_max() const
{
    if(!this->is_initialised())
        throw std::runtime_error("NiftiImageData::get_max(): Image not initialised.");

    // Get data
    return *std::max_element(_data, _data + _nifti_image->nvox);
}

float NiftiImageData::get_min() const
{
    if(!this->is_initialised())
        throw std::runtime_error("NiftiImageData::get_min(): Image not initialised.");

    // Get data
    return *std::min_element(_data, _data + _nifti_image->nvox);
}


float NiftiImageData::get_mean() const
{
    if(!this->is_initialised())
        throw std::runtime_error("NiftiImageData::get_min(): Image not initialised.");

    float sum = 0.F;
    int nan_count = 0;
    for (int i=0; i<int(_nifti_image->nvox); ++i)
        if (!isnan(_data[i])) {
            sum += _data[i];
            ++nan_count;
        }

    // Get data
    return sum / float(nan_count);
}

float NiftiImageData::get_sum() const
{
    if(!this->is_initialised())
        throw std::runtime_error("NiftiImageData::get_sum(): Image not initialised.");

    float sum = 0.F;
    for (unsigned i=0; i<_nifti_image->nvox; ++i)
        sum += float(_data[i]);
    return sum;
}

void NiftiImageData::fill(const float v)
{
    if(!this->is_initialised())
        throw std::runtime_error("NiftiImageData::fill(): Image not initialised.");

    for (unsigned i=0; i<_nifti_image->nvox; ++i)
        _data[i] = v;
}

float NiftiImageData::get_norm(const NiftiImageData& other) const
{
    if (!this->is_initialised())
        throw std::runtime_error("NiftiImageData::get_norm: first image is not initialised.");
    if (!other.is_initialised())
        throw std::runtime_error("NiftiImageData::get_norm: second image is not initialised.");
    for (int i=0; i<8; ++i)
        if (_nifti_image->dim[i] != other._nifti_image->dim[i])
            throw std::runtime_error("NiftiImageData::get_norm: dimensions do not match.");

    // Use double precision to minimise rounding errors
    double result(0);
    for (int i=0; i<int(_nifti_image->nvox); ++i) {
        const float &val1 = this->operator()(i);
        const float &val2 = other(i);
        // If either value is nan, skip
        if (!isnan(val1+val2))
            result += double(pow( this->operator()(i) - other(i), 2));
    }
    return float(sqrt(result));
}

NiftiImageData NiftiImageData::deep_copy() const
{
    NiftiImageData copy;
    copy = *this;
    return copy;
}

const int* NiftiImageData::get_dimensions() const
{
    return _nifti_image->dim;
}

void NiftiImageData::check_dimensions(const NiftiImageDataType image_type)
{
    if (!this->is_initialised())
        throw std::runtime_error("NiftiImageData::check_dimensions(): Image not initialised.");

    int ndim, nt, nu, intent_code, intent_p1;
    if        (image_type == _general)  { ndim=-1; nt=-1; nu=-1; intent_code = NIFTI_INTENT_NONE;   intent_p1=-1;         }
    else   if (image_type == _3D)       { ndim= 3; nt= 1; nu= 1; intent_code = NIFTI_INTENT_NONE;   intent_p1=-1;         }
    else   if (image_type == _3DTensor) { ndim= 5; nt= 1; nu= 3; intent_code = NIFTI_INTENT_VECTOR; intent_p1=-1;         }
    else   if (image_type == _3DDisp)   { ndim= 5; nt= 1; nu= 3; intent_code = NIFTI_INTENT_VECTOR; intent_p1=DISP_FIELD; }
    else /*if (image_type == _3DDef)*/  { ndim= 5; nt= 1; nu= 3; intent_code = NIFTI_INTENT_VECTOR; intent_p1=DEF_FIELD;  }

    // Check everthing is as it should be. -1 means we don't care about it
    // (e.g., NiftiImageData3D doesn't care about intent_p1, which is used by NiftyReg for Disp/Def fields)
    bool everything_ok = true;
    if (ndim         != -1 && ndim        != _nifti_image->ndim)        everything_ok = false;
    if ( nu          != -1 && nu          != _nifti_image->nu  )        everything_ok = false;
    if ( nt          != -1 && nt          != _nifti_image->nt  )        everything_ok = false;
    if ( intent_code != -1 && intent_code != _nifti_image->intent_code) everything_ok = false;
    if ( intent_p1   != -1 && intent_p1   != _nifti_image->intent_p1)   everything_ok = false;

    if (everything_ok)
        return;

    // If not, throw an error.
    std::stringstream ss;
    ss << "Trying to construct a ";
    if      (typeid(*this) == typeid(NiftiImageData3D))             ss << "NiftiImageData3D";
    else if (typeid(*this) == typeid(NiftiImageData3DTensor))       ss << "NiftiImageData3DTensor";
    else if (typeid(*this) == typeid(NiftiImageData3DDisplacement)) ss << "NiftiImageData3DDisplacement";
    else if (typeid(*this) == typeid(NiftiImageData3DDeformation))  ss << "NiftiImageData3DDeformation";
    ss << ".\n\t\tExpected params: ndim = " << ndim << ", nu = " << nu << ", nt = " << nt;
    if      (intent_code == NIFTI_INTENT_NONE)   ss << ", intent_code = None";
    else if (intent_code == NIFTI_INTENT_VECTOR) ss << ", intent_code = Vector";
    if      (intent_p1 == 0) ss << ", intent_p1 = Deformation";
    else if (intent_p1 == 1) ss << ", intent_p1 = Displacement";
    ss << "\n\t\tActual params:   ndim = " << _nifti_image->ndim << ", nu = " << _nifti_image->nu << ", nt = " << _nifti_image->nt;
    if      (_nifti_image->intent_code == NIFTI_INTENT_NONE)   ss << ", intent_code = None";
    else if (_nifti_image->intent_code == NIFTI_INTENT_VECTOR) ss << ", intent_code = Vector";
    if      (intent_p1 != -1 && _nifti_image->intent_p1 == 0)  ss << ", intent_p1 = Deformation";
    else if (intent_p1 != -1 && _nifti_image->intent_p1 == 1)  ss << ", intent_p1 = Displacement";
    //std::cout << ss.str() << "\n";
    throw std::runtime_error(ss.str());
}

NiftiImageData NiftiImageData::maths(const NiftiImageData& c, const MathsType type) const
{
    if (!this->is_initialised() || !c.is_initialised())
        throw std::runtime_error("NiftiImageData::maths_image_image: at least one image is not initialised.");
    if (!SIRFRegMisc::do_nifti_image_metadata_match(*this, c))
        throw std::runtime_error("NiftiImageData::maths_image_image: metadata do not match.");
    if (type != add && type != sub)
        throw std::runtime_error("NiftiImageData::maths_image_image: only implemented for add and subtract.");

    NiftiImageData res = this->deep_copy();

    for (int i=0; i<int(this->_nifti_image->nvox); ++i) {
        if (type == add) res(i) += c(i);
        else             res(i) -= c(i);
    }

    return res;
}

NiftiImageData NiftiImageData::maths(const float val, const MathsType type) const
{
    if (!this->is_initialised())
        throw std::runtime_error("NiftiImageData::maths_image_val: image is not initialised.");
    if (type != add && type != sub && type != mul)
        throw std::runtime_error("NiftiImageData::maths_image_val: only implemented for add, subtract and multiply.");

    NiftiImageData res = this->deep_copy();
    for (int i=0; i<int(this->_nifti_image->nvox); ++i) {
        if      (type == add) res(i) += val;
        else if (type == sub) res(i) -= val;
        else                  res(i) *= val;
    }
    return res;
}

void NiftiImageData::change_datatype(const int datatype)
{
    if      (datatype == DT_BINARY)   SIRFRegMisc::change_datatype<bool>(*this);
    else if (datatype == DT_INT8)     SIRFRegMisc::change_datatype<signed char>(*this);
    else if (datatype == DT_INT16)    SIRFRegMisc::change_datatype<signed short>(*this);
    else if (datatype == DT_INT32)    SIRFRegMisc::change_datatype<signed int>(*this);
    else if (datatype == DT_FLOAT32)  SIRFRegMisc::change_datatype<float>(*this);
    else if (datatype == DT_FLOAT64)  SIRFRegMisc::change_datatype<double>(*this);
    else if (datatype == DT_UINT8)    SIRFRegMisc::change_datatype<unsigned char>(*this);
    else if (datatype == DT_UINT16)   SIRFRegMisc::change_datatype<unsigned short>(*this);
    else if (datatype == DT_UINT32)   SIRFRegMisc::change_datatype<unsigned int>(*this);
    else if (datatype == DT_INT64)    SIRFRegMisc::change_datatype<signed long long>(*this);
    else if (datatype == DT_UINT64)   SIRFRegMisc::change_datatype<unsigned long long>(*this);
    else if (datatype == DT_FLOAT128) SIRFRegMisc::change_datatype<long double>(*this);
    else
        throw std::runtime_error("Trying to change to bad/unsupported datatype (" + std::to_string(datatype) + " / " + nifti_datatype_to_string(datatype) + ").");
}

/// Dump header info
void NiftiImageData::print_header() const
{
    NiftiImageData::print_headers({*this});
}

/// Dump multiple header info
void NiftiImageData::print_headers(const std::vector<NiftiImageData> &ims)
{
    SIRFRegMisc::dump_headers(ims);
}

void NiftiImageData::crop(const int min_index[7], const int max_index[7])
{
    if(!this->is_initialised())
        throw std::runtime_error("NiftiImageData::crop: Image not initialised.");

    std::shared_ptr<nifti_image> im = _nifti_image;

    // If any min values are -1, set them to 0. If any max values are -1, set them to dim[i]-1
    int min_idx[7], max_idx[7];
    for (int i=0; i<7; ++i) {
        (min_index[i] > -1) ? min_idx[i] = min_index[i] : min_idx[i] = 0;
        (max_index[i] > -1) ? max_idx[i] = max_index[i] : max_idx[i] = im->dim[i+1]-1;
    }

    // Check the min. and max. indices are in bounds.
    // Check the max. is less than the min.
    bool bounds_ok = true;
    if (!this->is_in_bounds(min_idx))  bounds_ok = false;
    if (!this->is_in_bounds(max_idx))  bounds_ok = false;
    for (int i=0; i<7; ++i)
        if (max_idx[i] > im->dim[i+1]) bounds_ok = false;
    if (!bounds_ok) {
        std::stringstream ss;
        ss << "crop_image: Bounds not ok.\n";
        ss << "\tImage dims              = (";
        for (int i=1; i<8; ++i) ss << im->dim[i] << " ";
        ss << ").\n\tMinimum requested index = (";
        for (int i=0; i<7; ++i) ss << min_idx[i] << " ";
        ss << ").\n\tMaximum requested index = (";
        for (int i=0; i<7; ++i) ss << max_idx[i] << " ";
        ss << ").\n";
        throw std::runtime_error(ss.str());
    }

    // Copy the original array
    const NiftiImageData copy = this->deep_copy();

    // Set the new number of voxels
    im->dim[1] = im->nx = max_idx[0] - min_idx[0] + 1;
    im->dim[2] = im->ny = max_idx[1] - min_idx[1] + 1;
    im->dim[3] = im->nz = max_idx[2] - min_idx[2] + 1;
    im->dim[4] = im->nt = max_idx[3] - min_idx[3] + 1;
    im->dim[5] = im->nu = max_idx[4] - min_idx[4] + 1;
    im->dim[6] = im->nv = max_idx[5] - min_idx[5] + 1;
    im->dim[7] = im->nw = max_idx[6] - min_idx[6] + 1;
    im->nvox = unsigned(im->nx * im->ny * im->nz * im->nt * im->nu * im->nv * im->nw);

    // Set the number of dimensions (if it's been decreased)
    if (im->dim[0] == 7 && im->nw == 1) im->dim[0] = im->ndim = 6;
    if (im->dim[0] == 6 && im->nv == 1) im->dim[0] = im->ndim = 5;
    if (im->dim[0] == 5 && im->nu == 1) im->dim[0] = im->ndim = 4;
    if (im->dim[0] == 4 && im->nt == 1) im->dim[0] = im->ndim = 3;
    if (im->dim[0] == 3 && im->nz == 1) im->dim[0] = im->ndim = 2;
    if (im->dim[0] == 2 && im->ny == 1) im->dim[0] = im->ndim = 1;
    if (im->dim[0] == 1 && im->nx == 1) im->dim[0] = im->ndim = 0;

    // Reset the data to the correct num of voxels
    free(im->data);
    im->data = static_cast<void*>(calloc(im->nvox,size_t(im->nbyper)));
    _data    = static_cast<float*>(im->data);

    // Get the data
    float *old_data = static_cast<float*>(copy.get_raw_nifti_sptr()->data);
    float *new_data = _data;

    int new_index[7], old_index[7];
    int new_1d_idx, old_1d_idx;

    // Fill the data
    for (old_index[6]=min_idx[6]; old_index[6]<=max_idx[6]; ++old_index[6]) {
        for (old_index[5]=min_idx[5]; old_index[5]<=max_idx[5]; ++old_index[5]) {
            for (old_index[4]=min_idx[4]; old_index[4]<=max_idx[4]; ++old_index[4]) {
                for (old_index[3]=min_idx[3]; old_index[3]<=max_idx[3]; ++old_index[3]) {
                    for (old_index[2]=min_idx[2]; old_index[2]<=max_idx[2]; ++old_index[2]) {
                        for (old_index[1]=min_idx[1]; old_index[1]<=max_idx[1]; ++old_index[1]) {
                            for (old_index[0]=min_idx[0]; old_index[0]<=max_idx[0]; ++old_index[0]) {

                                for (int i=0; i<7; ++i)
                                    new_index[i] = old_index[i] - min_idx[i];

                                new_1d_idx = this->get_1D_index(new_index);
                                old_1d_idx = copy.get_1D_index(old_index);
                                new_data[new_1d_idx] = old_data[old_1d_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    // If the minimum has been changed, need to alter the origin.
    for (int i=0; i<3; ++i)
        _nifti_image->qto_ijk.m[i][3] -= min_idx[i];
    _nifti_image->qto_xyz =
            nifti_mat44_inverse(_nifti_image->qto_ijk);
    nifti_mat44_to_quatern( _nifti_image->qto_xyz,
                            &_nifti_image->quatern_b,
                            &_nifti_image->quatern_c,
                            &_nifti_image->quatern_d,
                            &_nifti_image->qoffset_x,
                            &_nifti_image->qoffset_y,
                            &_nifti_image->qoffset_z,
                            nullptr,
                            nullptr,
                            nullptr,
                            &_nifti_image->qfac );
    _nifti_image->pixdim[0]=_nifti_image->qfac;
}

int NiftiImageData::get_1D_index(const int idx[7]) const
{
    // Get dims and spacing
    int *dim = _nifti_image->dim;

    // Check it's in bounds
    for (int i=0; i<7; ++i) {
        if (idx[i]<0 || idx[i]>=dim[i+1]) {
            std::stringstream ss;
            ss << "NiftiImageData::get_1D_index: Element out of bounds.\n";
            ss << "\tRequested = ( ";
            for (int i=0;i<7;++i) ss << idx[i] << " ";
            ss << ")\n\tBounds    = ( ";
            for (int i=0;i<7;++i) ss << dim[i+1] << " ";
            ss << ")";
            throw std::runtime_error(ss.str());
        }
    }

    int idx_1d = 0;
    idx_1d += idx[0];
    idx_1d += idx[1] * dim[1];
    idx_1d += idx[2] * dim[1] * dim[2];
    idx_1d += idx[3] * dim[1] * dim[2] * dim[3];
    idx_1d += idx[4] * dim[1] * dim[2] * dim[3] * dim[4];
    idx_1d += idx[5] * dim[1] * dim[2] * dim[3] * dim[4] * dim[5];
    idx_1d += idx[6] * dim[1] * dim[2] * dim[3] * dim[4] * dim[5] * dim[6];

    return idx_1d;
}

void NiftiImageData::set_up_data(const int original_datatype)
{
    // Save the original datatype, we'll convert it back to this just before saving
    _original_datatype = original_datatype;

    // TODO display a warning that data will be lost if original was e.g., double
    if (original_datatype != NIFTI_TYPE_FLOAT32) {
        if (_nifti_image->nbyper > int(sizeof(float)))
            std::cout << "\nDecreasing number of bytes per pixel, could cause loss of accuracy.\n"
                 << "Input data type was " << nifti_datatype_to_string(original_datatype)
                 << ", converting to " << nifti_datatype_to_string(NIFTI_TYPE_FLOAT32) << ".\n";

        this->change_datatype(NIFTI_TYPE_FLOAT32);
    }

    _nifti_image->nbyper = sizeof(float);
    this->_data = static_cast<float*>(_nifti_image->data);
}

bool NiftiImageData::is_in_bounds(const int index[7]) const
{
    for (int i=0; i<7; ++i)
        if (index[i]<0 || index[0]>=_nifti_image->dim[1])
            return false;
    return true;
}

bool NiftiImageData::is_in_bounds(const int index) const
{
    return (index>=0 && index<int(_nifti_image->nvox));
}

bool NiftiImageData::is_same_size(const NiftiImageData &im) const
{
    for (int i=0; i<8; ++i)
        if (_nifti_image->dim[i] != im._nifti_image->dim[i])
            return false;
    return true;
}

bool NiftiImageData::are_equal_to_given_accuracy(const NiftiImageData &im2, const float required_accuracy_compared_to_max) const
{
    const NiftiImageData &im1 = *this;

    if(!im1.is_initialised())
        throw std::runtime_error("NiftiImageData::are_equal_to_given_accuracy: Image 1 not initialised.");
    if(!im2.is_initialised())
        throw std::runtime_error("NiftiImageData::are_equal_to_given_accuracy: Image 2 not initialised.");

    // Get norm between two images
    float norm = im1.get_norm(im2);
    float epsilon = (im1.get_max()+im2.get_max())/2.F;
    epsilon *= required_accuracy_compared_to_max;

    if (norm < epsilon)
        return true;

    std::cout << "\nImages are not equal (norm > epsilon).\n";
    std::cout << "\tmax1                              = " << im1.get_max() << "\n";
    std::cout << "\tmax2                              = " << im1.get_max() << "\n";
    std::cout << "\tmin1                              = " << im1.get_min() << "\n";
    std::cout << "\tmin2                              = " << im2.get_min() << "\n";
    std::cout << "\trequired accuracy compared to max = " << required_accuracy_compared_to_max << "\n";
    std::cout << "\tepsilon                           = " << epsilon << "\n";
    std::cout << "\tnorm                              = " << norm << "\n";
    return false;
}
