/*
CCP PETMR Synergistic Image Reconstruction Framework (SIRF)
Copyright 2015 - 2017 Rutherford Appleton Laboratory STFC
Copyright 2015 - 2017 University College London.
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
#define CSIRFREG_FOR_MATLAB
#ifdef _WIN32
#define EXPORTED_FUNCTION __declspec(dllexport)
#else
#define EXPORTED_FUNCTION
#endif

#include <mex.h>
#include "matrix.h"
#include "csirfreg.h"

#ifndef CSIRFREG_FOR_MATLAB
#define PTR_INT size_t
#define PTR_FLOAT size_t
#define PTR_DOUBLE size_t
 extern "C" {
#else
#define PTR_INT int*
#define PTR_FLOAT float*
#define PTR_DOUBLE double*
#endif
EXPORTED_FUNCTION  void* mSIRFReg_newObject(const char* name) {
	return cSIRFReg_newObject(name);
}
EXPORTED_FUNCTION 	void* mSIRFReg_objectFromFile(const char* name, const char* filename) {
	return cSIRFReg_objectFromFile(name, filename);
}
EXPORTED_FUNCTION 	void* mSIRFReg_setParameter (void* ptr, const char* obj, const char* name, const void* value) {
	return cSIRFReg_setParameter (ptr, obj, name, value);
}
EXPORTED_FUNCTION 	void* mSIRFReg_parameter(const void* ptr, const char* obj, const char* name) {
	return cSIRFReg_parameter(ptr, obj, name);
}
EXPORTED_FUNCTION     void* mSIRFReg_NiftiImageData_print_headers(const int num_ims, const void* im1, const void* im2, const void* im3, const void* im4, const void* im5) {
	return cSIRFReg_NiftiImageData_print_headers(num_ims, im1, im2, im3, im4, im5);
}
EXPORTED_FUNCTION     void* mSIRFReg_NiftiImageData_save_to_file(const void* ptr, const char* filename, const int datatype) {
	return cSIRFReg_NiftiImageData_save_to_file(ptr, filename, datatype);
}
EXPORTED_FUNCTION     void* mSIRFReg_NiftiImageData_fill(const void* ptr, const float val) {
	return cSIRFReg_NiftiImageData_fill(ptr, val);
}
EXPORTED_FUNCTION     void* mSIRFReg_NiftiImageData_fill_arr(const void* ptr, PTR_FLOAT val) {
	return cSIRFReg_NiftiImageData_fill_arr(ptr, val);
}
EXPORTED_FUNCTION     void* mSIRFReg_NiftiImageData_deep_copy(const void* copy_ptr, const void *orig_ptr) {
	return cSIRFReg_NiftiImageData_deep_copy(copy_ptr, orig_ptr);
}
EXPORTED_FUNCTION     void* mSIRFReg_NiftiImageData_get_dimensions(const void* ptr, PTR_INT ptr_dim) {
	return cSIRFReg_NiftiImageData_get_dimensions(ptr, ptr_dim);
}
EXPORTED_FUNCTION     void* mSIRFReg_NiftiImageData_get_data(const void* ptr, PTR_FLOAT ptr_data) {
	return cSIRFReg_NiftiImageData_get_data(ptr, ptr_data);
}
EXPORTED_FUNCTION     void* mSIRFReg_NiftiImageData_maths_im(const void *res_ptr, const void* im1_ptr, const void* im2_ptr, const int maths_type) {
	return cSIRFReg_NiftiImageData_maths_im(res_ptr, im1_ptr, im2_ptr, maths_type);
}
EXPORTED_FUNCTION     void* mSIRFReg_NiftiImageData_maths_num(const void *res_ptr, const void* im1_ptr, const float val, const int maths_type) {
	return cSIRFReg_NiftiImageData_maths_num(res_ptr, im1_ptr, val, maths_type);
}
EXPORTED_FUNCTION     void* mSIRFReg_NiftiImageData_equal(const void* im1_ptr, const void* im2_ptr) {
	return cSIRFReg_NiftiImageData_equal(im1_ptr, im2_ptr);
}
EXPORTED_FUNCTION     void* mSIRFReg_NiftiImageData_norm(const void* im1_ptr, const void* im2_ptr) {
	return cSIRFReg_NiftiImageData_norm(im1_ptr, im2_ptr);
}
EXPORTED_FUNCTION     void* mSIRFReg_NiftiImageData_get_original_datatype(const void* im_ptr) {
	return cSIRFReg_NiftiImageData_get_original_datatype(im_ptr);
}
EXPORTED_FUNCTION     void* mSIRFReg_NiftiImageData_crop(const void* im_ptr, PTR_INT min_index_ptr, PTR_INT max_index_ptr) {
	return cSIRFReg_NiftiImageData_crop(im_ptr, min_index_ptr, max_index_ptr);
}
EXPORTED_FUNCTION     void* mSIRFReg_NiftiImageData3D_from_PETImageData(void* ptr) {
	return cSIRFReg_NiftiImageData3D_from_PETImageData(ptr);
}
EXPORTED_FUNCTION     void* mSIRFReg_NiftiImageData3D_copy_data_to(const void* ptr, const void* obj) {
	return cSIRFReg_NiftiImageData3D_copy_data_to(ptr, obj);
}
EXPORTED_FUNCTION     void* mSIRFReg_NiftiImageData3DTensor_save_to_file_split_xyz_components(const void* ptr, const char* filename, const int datatype) {
	return cSIRFReg_NiftiImageData3DTensor_save_to_file_split_xyz_components(ptr, filename, datatype);
}
EXPORTED_FUNCTION     void* mSIRFReg_NiftiImageData3DTensor_create_from_3D_image(const void *ptr, const void* obj) {
	return cSIRFReg_NiftiImageData3DTensor_create_from_3D_image(ptr, obj);
}
EXPORTED_FUNCTION     void* mSIRFReg_NiftiImageData3DTensor_construct_from_3_components(const char* obj, const void *x_ptr, const void *y_ptr, const void *z_ptr) {
	return cSIRFReg_NiftiImageData3DTensor_construct_from_3_components(obj, x_ptr, y_ptr, z_ptr);
}
EXPORTED_FUNCTION     void* mSIRFReg_NiftiImageData3DTensor_flip_component(const void *ptr, const int dim) {
	return cSIRFReg_NiftiImageData3DTensor_flip_component(ptr, dim);
}
EXPORTED_FUNCTION     void* mSIRFReg_NiftiImageData3DDeformation_compose_single_deformation(const void* im, const int num_elements, const char* types, const void* trans1, const void* trans2, const void* trans3, const void* trans4, const void* trans5) {
	return cSIRFReg_NiftiImageData3DDeformation_compose_single_deformation(im, num_elements, types, trans1, trans2, trans3, trans4, trans5);
}
EXPORTED_FUNCTION     void* mSIRFReg_NiftiImageData3DDeformation_create_from_disp(const void* ptr, const void* disp_ptr) {
	return cSIRFReg_NiftiImageData3DDeformation_create_from_disp(ptr, disp_ptr);
}
EXPORTED_FUNCTION     void* mSIRFReg_NiftiImageData3DDisplacement_create_from_def(const void* ptr, const void* def_ptr) {
	return cSIRFReg_NiftiImageData3DDisplacement_create_from_def(ptr, def_ptr);
}
EXPORTED_FUNCTION     void* mSIRFReg_SIRFReg_process(void* ptr) {
	return cSIRFReg_SIRFReg_process(ptr);
}
EXPORTED_FUNCTION     void* mSIRFReg_SIRFReg_get_deformation_displacement_image(const void* ptr, const char *transform_type) {
	return cSIRFReg_SIRFReg_get_deformation_displacement_image(ptr, transform_type);
}
EXPORTED_FUNCTION     void* mSIRFReg_SIRFReg_set_parameter(const void* ptr, const char* par, const char* arg1, const char* arg2) {
	return cSIRFReg_SIRFReg_set_parameter(ptr, par, arg1, arg2);
}
EXPORTED_FUNCTION     void* mSIRFReg_SIRFReg_get_TM(const void* ptr, const char* dir) {
	return cSIRFReg_SIRFReg_get_TM(ptr, dir);
}
EXPORTED_FUNCTION     void* mSIRFReg_SIRFRegNiftyResample_add_transformation(void* self, const void* trans, const char* type) {
	return cSIRFReg_SIRFRegNiftyResample_add_transformation(self, trans, type);
}
EXPORTED_FUNCTION     void* mSIRFReg_SIRFRegNiftyResample_process(void* ptr) {
	return cSIRFReg_SIRFRegNiftyResample_process(ptr);
}
EXPORTED_FUNCTION     void* mSIRFReg_SIRFRegImageWeightedMean_add_image(void* ptr, const void* obj, const float weight) {
	return cSIRFReg_SIRFRegImageWeightedMean_add_image(ptr, obj, weight);
}
EXPORTED_FUNCTION     void* mSIRFReg_SIRFRegImageWeightedMean_add_image_filename(void* ptr, const char* filename, const float weight) {
	return cSIRFReg_SIRFRegImageWeightedMean_add_image_filename(ptr, filename, weight);
}
EXPORTED_FUNCTION     void* mSIRFReg_SIRFRegImageWeightedMean_process(void* ptr) {
	return cSIRFReg_SIRFRegImageWeightedMean_process(ptr);
}
EXPORTED_FUNCTION     void* mSIRFReg_SIRFRegTransformation_get_as_deformation_field(const void* ptr, const char* name, const void* ref) {
	return cSIRFReg_SIRFRegTransformation_get_as_deformation_field(ptr, name, ref);
}
EXPORTED_FUNCTION     void* mSIRFReg_SIRFRegAffineTransformation_construct_from_TM(PTR_FLOAT ptr_TM) {
	return cSIRFReg_SIRFRegAffineTransformation_construct_from_TM(ptr_TM);
}
EXPORTED_FUNCTION     void* mSIRFReg_SIRFRegAffineTransformation_deep_copy(const void* ptr) {
	return cSIRFReg_SIRFRegAffineTransformation_deep_copy(ptr);
}
EXPORTED_FUNCTION     void* mSIRFReg_SIRFRegAffineTransformation_save_to_file(const void* ptr, const char* filename) {
	return cSIRFReg_SIRFRegAffineTransformation_save_to_file(ptr, filename);
}
EXPORTED_FUNCTION     void* mSIRFReg_SIRFRegAffineTransformation_as_array(const void* ptr, PTR_FLOAT ptr_TM) {
	return cSIRFReg_SIRFRegAffineTransformation_as_array(ptr, ptr_TM);
}
EXPORTED_FUNCTION     void* mSIRFReg_SIRFRegAffineTransformation_get_identity() {
	return cSIRFReg_SIRFRegAffineTransformation_get_identity();
}
EXPORTED_FUNCTION     void* mSIRFReg_SIRFRegAffineTransformation_get_inverse(const void* ptr) {
	return cSIRFReg_SIRFRegAffineTransformation_get_inverse(ptr);
}
EXPORTED_FUNCTION     void* mSIRFReg_SIRFRegAffineTransformation_mul(const void* mat1_ptr, const void* mat2_ptr) {
	return cSIRFReg_SIRFRegAffineTransformation_mul(mat1_ptr, mat2_ptr);
}
EXPORTED_FUNCTION     void* mSIRFReg_SIRFRegAffineTransformation_equal(const void* mat1_ptr, const void* mat2_ptr) {
	return cSIRFReg_SIRFRegAffineTransformation_equal(mat1_ptr, mat2_ptr);
}
#ifndef CSIRFREG_FOR_MATLAB
}
#endif
void* newMexPrinter();
void* deleteMexPrinter(void* ptr);
EXPORTED_FUNCTION void* mNewMexPrinter() {
  return newMexPrinter();
}
EXPORTED_FUNCTION void* mDeleteMexPrinter(void* ptr) {
  return deleteMexPrinter(ptr);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {}
