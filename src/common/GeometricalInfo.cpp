/*
CCP PETMR Synergistic Image Reconstruction Framework (SIRF)
Copyright 2019 University College London.

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

#include "sirf/common/GeometricalInfo.h"
#include <iostream>
#include <math.h>

using namespace sirf;

template <int num_dimensions>
void
VoxelisedGeometricalInfo<num_dimensions>::
print_info() const
{
    std::cout << "Offset: (";
    std::cout << _offset[0] << ", " << _offset[1] << ", " << _offset[2] << ")\n";

    std::cout << "Spacing: (";
    std::cout << _spacing[0] << ", " << _spacing[1] << ", " << _spacing[2] << ")\n";

    std::cout << "Size: (";
    std::cout << _size[0] << ", " << _size[1] << ", " << _size[2] << ")\n";

    std::cout << "Dir mat: \n";
    for( int i=0;i<3; i++) {
        for( int j=0;j<3; j++) {
            std::cout << _direction[i][j];
            if (j<2) std::cout << ", ";
            else     std::cout << "\n";
        }
    }
    std::cout << "\n";
}

template <int num_dimensions>
typename VoxelisedGeometricalInfo<num_dimensions>::Coordinate
VoxelisedGeometricalInfo<num_dimensions>::
multiply_by_direction_matrix(const DirectionMatrix &matrix, const Coordinate &input)
{
    Coordinate output = {0.f, 0.f, 0.f};
    for (unsigned i=0; i<3; ++i)
        for (unsigned j=0; j<3; ++j)
            output[i] += matrix[i][j] * input[j];
    return output;
}

template <int num_dimensions>
typename VoxelisedGeometricalInfo<num_dimensions>::Index
VoxelisedGeometricalInfo<num_dimensions>::
multiply_by_direction_matrix(const DirectionMatrix &matrix, const Index &input)
{
    Coordinate temp = {0.f, 0.f, 0.f};
    for (unsigned i=0; i<3; ++i)
        for (unsigned j=0; j<3; ++j)
            temp[i] += matrix[i][j] * input[j];
    Index output = { unsigned(temp[0]),unsigned(temp[1]),unsigned(temp[2]) };
    return output;
}

template <int num_dimensions>
typename VoxelisedGeometricalInfo<num_dimensions>::DirectionMatrix
VoxelisedGeometricalInfo<num_dimensions>::
multiply_direction_matrices(const DirectionMatrix &input1, const DirectionMatrix &input2)
{
    DirectionMatrix output;
    for (unsigned i=0; i<3; ++i)
        for (unsigned j=0; j<3; ++j)
            output[i][j] = 0.f;
    for (unsigned i=0; i<3; ++i)
        for (unsigned j=0; j<3; ++j)
            for (unsigned k=0; k<3; ++k)
                output[i][j] += input1[i][k] * input2[k][j];

    return output;
}

template <int num_dimensions>
typename VoxelisedGeometricalInfo<num_dimensions>::DirectionMatrix
VoxelisedGeometricalInfo<num_dimensions>::
inverse_direction_matrix(const DirectionMatrix &dm)
{
    DirectionMatrix output;

    // computes the inverse of a matrix m
    float det = dm[0][0] * (dm[1][1] * dm[2][2] - dm[2][1] * dm[1][2]) -
                dm[0][1] * (dm[1][0] * dm[2][2] - dm[1][2] * dm[2][0]) +
                dm[0][2] * (dm[1][0] * dm[2][1] - dm[1][1] * dm[2][0]);

    float invdet = 1.f / det;

    output[0][0] = (dm[1][1] * dm[2][2] - dm[2][1] * dm[1][2]) * invdet;
    output[0][1] = (dm[0][2] * dm[2][1] - dm[0][1] * dm[2][2]) * invdet;
    output[0][2] = (dm[0][1] * dm[1][2] - dm[0][2] * dm[1][1]) * invdet;
    output[1][0] = (dm[1][2] * dm[2][0] - dm[1][0] * dm[2][2]) * invdet;
    output[1][1] = (dm[0][0] * dm[2][2] - dm[0][2] * dm[2][0]) * invdet;
    output[1][2] = (dm[1][0] * dm[0][2] - dm[0][0] * dm[1][2]) * invdet;
    output[2][0] = (dm[1][0] * dm[2][1] - dm[2][0] * dm[1][1]) * invdet;
    output[2][1] = (dm[2][0] * dm[0][1] - dm[0][0] * dm[2][1]) * invdet;
    output[2][2] = (dm[0][0] * dm[1][1] - dm[1][0] * dm[0][1]) * invdet;

    return output;
}

template <int num_dimensions>
typename VoxelisedGeometricalInfo<num_dimensions>::DirectionMatrix
VoxelisedGeometricalInfo<num_dimensions>::
absolute_direction_matrix(const DirectionMatrix &dm)
{
    DirectionMatrix output;
    for (unsigned i=0; i<4; ++i)
        for (unsigned j=0; j<4; ++j)
            output[i][j] = fabs(dm[i][j]);
    return output;
}

template <int num_dimensions>
VoxelisedGeometricalInfo<num_dimensions>::
VoxelisedGeometricalInfo(
	const Offset& offset, const Spacing& spacing,
	const Size& size, const DirectionMatrix& direction)
	:
	_offset(offset),
	_spacing(spacing),
	_size(size),
	_direction(direction)
{}

template <int num_dimensions>
const typename VoxelisedGeometricalInfo<num_dimensions>::Offset
VoxelisedGeometricalInfo<num_dimensions>::get_offset() const
{
	return _offset;
}

template <int num_dimensions>
const typename VoxelisedGeometricalInfo<num_dimensions>::Spacing
VoxelisedGeometricalInfo<num_dimensions>::get_spacing() const
{
	return _spacing;
}

template <int num_dimensions>
const typename VoxelisedGeometricalInfo<num_dimensions>::Size
VoxelisedGeometricalInfo<num_dimensions>::get_size() const
{
	return _size;
}

template <int num_dimensions>
const typename VoxelisedGeometricalInfo<num_dimensions>::DirectionMatrix
VoxelisedGeometricalInfo<num_dimensions>::get_direction() const
{
	return _direction;
}

template <int num_dimensions>
const typename VoxelisedGeometricalInfo<num_dimensions>::TransformMatrix
VoxelisedGeometricalInfo<num_dimensions>::
calculate_index_to_physical_point_matrix() const
{
	TransformMatrix index_to_physical_point_matrix;
	for (unsigned int dim = 0; dim<num_dimensions; dim++) {
		for (unsigned int axis = 0; axis<num_dimensions; axis++) {
			index_to_physical_point_matrix[dim][axis] =
				_direction[dim][axis] * _spacing[axis];
		}
		index_to_physical_point_matrix[dim][num_dimensions] = _offset[dim];
		index_to_physical_point_matrix[num_dimensions][dim] = 0;
	}
	index_to_physical_point_matrix[num_dimensions][num_dimensions] = 1;
	return index_to_physical_point_matrix;
}

namespace sirf {
template class VoxelisedGeometricalInfo<3>;
}
