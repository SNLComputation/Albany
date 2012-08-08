/********************************************************************\
*            Albany, Copyright (2012) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Glen Hansen, gahanse@sandia.gov                    *
\********************************************************************/


#ifndef MULTISCALESTRESS_HPP
#define MULTISCALESTRESS_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace LCM {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

//! MPI message tags
enum MessageType {STRESS_TENSOR, STRAIN_TENSOR, TANGENT, DIE};


template<typename EvalT, typename Traits>
class MultiScaleStressBase : public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>  {

  public:

    MultiScaleStressBase(const Teuchos::ParameterList& p);

    void postRegistrationSetup(typename Traits::SetupData d,
                               PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);

  protected:

    void calcStress(typename Traits::EvalData workset);

    // Protected function for stress calc, only for RealType
    void mesoBridgeStressRealType(PHX::MDField<RealType, Cell, QuadPoint, Dim, Dim>& stressFieldOut,
                                  PHX::MDField<RealType, Cell, QuadPoint, Dim, Dim>& stressFieldIn,
                                  typename Traits::EvalData workset);

    PHX::MDField<RealType, Cell, QuadPoint, Dim, Dim> stressFieldRealType;


    struct MesoPt {

      int cell;
      int qp;

    };

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    // Input:
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> strain;
    PHX::MDField<ScalarT, Cell, QuadPoint> elasticModulus;
    PHX::MDField<ScalarT, Cell, QuadPoint> poissonsRatio;

    unsigned int numQPs;
    unsigned int numDims;

    int numMesoPEs;
    std::vector<double> exchanged_stresses;
    std::vector<MesoPt> loc_data;
    Teuchos::RCP<MPI_Comm> interCommunicator;


    // Output:
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> stress;

    // Link convenience functions

    void sendCellQPData(int cell, int qp, int toProc, MessageType type,
                        PHX::MDField<RealType, Cell, QuadPoint, Dim, Dim>& stressFieldIn);

    void rcvCellQPData(int procIDReached, MessageType type,
                       PHX::MDField<RealType, Cell, QuadPoint, Dim, Dim>& stressFieldOut);

};

// Inherted classes
template<typename EvalT, typename Traits> class MultiScaleStress;

// For all cases except those specialized below, just fall through to base class.
// The base class throws "Not Implemented" for evaluate fields.
template<typename EvalT, typename Traits>
class MultiScaleStress : public MultiScaleStressBase<EvalT, Traits> {
  public:
    MultiScaleStress(Teuchos::ParameterList& p) : MultiScaleStressBase<EvalT, Traits>(p) {};
};


// Template Specialization: Residual Eval calls MultiScale with doubles.
template<typename Traits>
class MultiScaleStress<PHAL::AlbanyTraits::Residual, Traits>
    : public MultiScaleStressBase<PHAL::AlbanyTraits::Residual, Traits> {
  public:
    MultiScaleStress(Teuchos::ParameterList& p) : MultiScaleStressBase<PHAL::AlbanyTraits::Residual, Traits>(p) {};
    void evaluateFields(typename Traits::EvalData d);
};

// Template Specialization: Jacobian Eval does finite difference of MultiScale with doubles.
template<typename Traits>
class MultiScaleStress<PHAL::AlbanyTraits::Jacobian, Traits>
    : public MultiScaleStressBase<PHAL::AlbanyTraits::Jacobian, Traits> {
  public:
    MultiScaleStress(Teuchos::ParameterList& p) : MultiScaleStressBase<PHAL::AlbanyTraits::Jacobian, Traits>(p) {};
    void evaluateFields(typename Traits::EvalData d);
};

// Template Specialization: Tangent Eval does finite difference of MultiScale with doubles.
template<typename Traits>
class MultiScaleStress<PHAL::AlbanyTraits::Tangent, Traits>
    : public MultiScaleStressBase<PHAL::AlbanyTraits::Tangent, Traits> {
  public:
    MultiScaleStress(Teuchos::ParameterList& p) : MultiScaleStressBase<PHAL::AlbanyTraits::Tangent, Traits>(p) {};
    void evaluateFields(typename Traits::EvalData d);
};

}

#endif
