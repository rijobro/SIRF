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

#pragma once

#include "sirf/Reg/NiftyRegistration.h"

template<class dataType> class reg_aladin_sym;

namespace sirf {

/// Forward declarations
template<class dataType> class AffineTransformation;

/*!
\ingroup Registration
\brief NiftyReg's aladin class for rigid and affine registrations.

Since this algorithm is affine/rigid, it can also return a transformation matrix if desired.

\author Richard Brown
\author CCP PETMR
*/
template<class dataType> class NiftyAladinSym : public NiftyRegistration<dataType>
{
public:

    /// Process
    void process();

    /// Get forwards transformation matrix
    const std::shared_ptr<const AffineTransformation<float> > get_transformation_matrix_forward_sptr() const { return _TM_forward_sptr; }

    /// Get inverse transformation matrix
    const std::shared_ptr<const AffineTransformation<float> > get_transformation_matrix_inverse_sptr() const { return _TM_inverse_sptr; }

    /// Print all wrapped methods
    static void print_all_wrapped_methods();

    void SetInterpolationToCubic           ();
    void SetInterpolationToNearestNeighbor ();
    void SetInterpolationToTrilinear       ();
    void SetAlignCentre                    (const bool         arg1);
    void SetPerformAffine                  (const bool         arg1);
    void SetPerformRigid                   (const bool         arg1);
    void SetVerbose                        (const bool         arg1);
    void SetBlockPercentage                (const int          arg1);
    void SetInterpolation                  (const int          arg1);
    void SetBlockStepSize                  (const int          arg1);
    void SetLevelsToPerform                (const unsigned     arg1);
    void SetMaxIterations                  (const unsigned     arg1);
    void SetNumberOfLevels                 (const unsigned     arg1);
    void SetFloatingSigma                  (const float        arg1);
    void SetInlierLts                      (const float        arg1);
    void SetReferenceSigma                 (const float        arg1);
    void SetFloatingLowerThreshold         (const float        arg1);
    void SetFloatingUpperThreshold         (const float        arg1);
    void SetReferenceLowerThreshold        (const float        arg1);
    void SetReferenceUpperThreshold        (const float        arg1);
    void SetWarpedPaddingValue             (const float        arg1);
    void SetAlignCentreMass                (const int          arg1);

protected:

    /// Parse parameter file
    virtual void parse_parameter_file();

    /// Set extra parameters.
    void set_parameters();

    /// Register object
    std::shared_ptr<reg_aladin_sym<dataType> > _registration_sptr;

    /// Forwards transformation matrix
    std::shared_ptr<AffineTransformation<float> > _TM_forward_sptr;
    /// Inverse transformation matrix
    std::shared_ptr<AffineTransformation<float> > _TM_inverse_sptr;
};
}
