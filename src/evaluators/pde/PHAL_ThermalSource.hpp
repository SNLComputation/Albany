//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_THERMALSOURCE_HPP
#define PHAL_THERMALSOURCE_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace PHAL {

/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class ThermalSource : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  ThermalSource(Teuchos::ParameterList const& p);

  void
  postRegistrationSetup(
      typename Traits::SetupData d,
      PHX::FieldManager<Traits>& vm);

  void
  evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT     ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  Teuchos::Array<double> kappa;  // Thermal Conductivity array
  double                 C;      // Heat Capacity
  double                 rho;    // Density
  PHX::MDField<const MeshScalarT, Cell, QuadPoint, Dim> coordVec;

  // Output:
  PHX::MDField<ScalarT, Cell, QuadPoint> Source;

  unsigned int numQPs, numDims, numNodes;
  
  enum FTYPE {NONE};
  FTYPE force_type;
};
}

#endif
