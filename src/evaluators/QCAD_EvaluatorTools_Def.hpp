//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"

//#define MESH_SCALAR_IS_AD_TYPE

// **********************************************************************
//   RESIDUAL
// **********************************************************************

template<typename Traits>
QCAD::EvaluatorTools<PHAL::AlbanyTraits::Residual,Traits>::
EvaluatorTools()
{
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::Residual, Traits>::
getDoubleValue(const ScalarT& t) const
{
  return t;
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::Residual, Traits>::
getMeshDoubleValue(const MeshScalarT& t) const
{
  return t;
}

template<typename Traits>
std::string QCAD::EvaluatorTools<PHAL::AlbanyTraits::Residual, Traits>::
getEvalType() const
{
  return "Residual";
}


// **********************************************************************
//   JACOBIAN
// **********************************************************************

template<typename Traits>
QCAD::EvaluatorTools<PHAL::AlbanyTraits::Jacobian,Traits>::
EvaluatorTools()
{
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::Jacobian, Traits>::
getDoubleValue(const ScalarT& t) const
{
  return t.val();
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::Jacobian, Traits>::
getMeshDoubleValue(const MeshScalarT& t) const
{
  return Albany::ADValue(t);
 //#ifdef MESH_SCALAR_IS_AD_TYPE
 //   return t.val();
 // #else
 //   return t;
 // #endif
}

template<typename Traits>
std::string QCAD::EvaluatorTools<PHAL::AlbanyTraits::Jacobian, Traits>::
getEvalType() const
{
  return "Jacobian";
}



// **********************************************************************
//   TANGENT
// **********************************************************************

template<typename Traits>
QCAD::EvaluatorTools<PHAL::AlbanyTraits::Tangent,Traits>::
EvaluatorTools()
{
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::Tangent, Traits>::
getDoubleValue(const ScalarT& t) const
{
  return t.val();
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::Tangent, Traits>::
getMeshDoubleValue(const MeshScalarT& t) const
{
  return Albany::ADValue(t);
  //#ifdef MESH_SCALAR_IS_AD_TYPE
  // return t.val();
  //#else
  //  return 0.0;
    //return t;  //Error about mesh scalar type not convertable to double -- EGN needs to talk with Andy here.
  //#endif
}

template<typename Traits>
std::string QCAD::EvaluatorTools<PHAL::AlbanyTraits::Tangent, Traits>::
getEvalType() const
{
  return "Tangent";
}

// **********************************************************************
//   DISTRIBUTED PARAMETER DERIVATIVE
// **********************************************************************

template<typename Traits>
QCAD::EvaluatorTools<PHAL::AlbanyTraits::DistParamDeriv,Traits>::
EvaluatorTools()
{
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
getDoubleValue(const ScalarT& t) const
{
  return t.val();
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
getMeshDoubleValue(const MeshScalarT& t) const
{
  return Albany::ADValue(t);
  //#ifdef MESH_SCALAR_IS_AD_TYPE
  //  return t.val();
  //#else
  //  return 0.0;
    //return t;  //Error about mesh scalar type not convertable to double -- EGN needs to talk with Andy here.
  //#endif
}

template<typename Traits>
std::string QCAD::EvaluatorTools<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
getEvalType() const
{
  return "DistParamDeriv";
}

#ifdef ALBANY_ENSEMBLE 

// **********************************************************************
//   MULTI-POINT RESIDUAL
// **********************************************************************

template<typename Traits>
QCAD::EvaluatorTools<PHAL::AlbanyTraits::MPResidual,Traits>::
EvaluatorTools()
{
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::MPResidual, Traits>::
getDoubleValue(const ScalarT& t) const
{
  return t.val(); 
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::MPResidual, Traits>::
getMeshDoubleValue(const MeshScalarT& t) const
{
  return Albany::ADValue(t);
  //#ifdef MESH_SCALAR_IS_AD_TYPE
  //  return t.val();
  //#else
  //  return t;
  //#endif
}

template<typename Traits>
std::string QCAD::EvaluatorTools<PHAL::AlbanyTraits::MPResidual, Traits>::
getEvalType() const
{
  return "MPResidual";
}


// **********************************************************************
//   MULTI-POINT JACOBIAN
// **********************************************************************

template<typename Traits>
QCAD::EvaluatorTools<PHAL::AlbanyTraits::MPJacobian,Traits>::
EvaluatorTools()
{
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::MPJacobian, Traits>::
getDoubleValue(const ScalarT& t) const
{
  return t.val().val(); 
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::MPJacobian, Traits>::
getMeshDoubleValue(const MeshScalarT& t) const
{
  return Albany::ADValue(t);
  //#ifdef MESH_SCALAR_IS_AD_TYPE
  //  return t.val().val();
  //#else
  //  return t;
  //#endif
}

template<typename Traits>
std::string QCAD::EvaluatorTools<PHAL::AlbanyTraits::MPJacobian, Traits>::
getEvalType() const
{
  return "MPJacobian";
}


// **********************************************************************
//   MULTI-POINT TANGENT
// **********************************************************************

template<typename Traits>
QCAD::EvaluatorTools<PHAL::AlbanyTraits::MPTangent,Traits>::
EvaluatorTools()
{
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::MPTangent, Traits>::
getDoubleValue(const ScalarT& t) const
{
  return t.val().val();
}

template<typename Traits>
double QCAD::EvaluatorTools<PHAL::AlbanyTraits::MPTangent, Traits>::
getMeshDoubleValue(const MeshScalarT& t) const
{
  return Albany::ADValue(t);
 // #ifdef MESH_SCALAR_IS_AD_TYPE
 //   return t.val().val();
 // #else
 //   return t;
 // #endif
}

template<typename Traits>
std::string QCAD::EvaluatorTools<PHAL::AlbanyTraits::MPTangent, Traits>::
getEvalType() const
{
  return "MPTangent";
}

#endif

// **********************************************************************

