//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"

#include "LandIce_HydraulicPotential.hpp"

#include "Albany_DiscretizationUtils.hpp"

namespace LandIce {

template<typename EvalT, typename Traits>
HydraulicPotential<EvalT, Traits>::
HydraulicPotential (const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl)
{
  // Check if it is a sideset evaluation
  eval_on_side = false;
  if (p.isParameter("Side Set Name")) {
    sideSetName = p.get<std::string>("Side Set Name");
    eval_on_side = true;
  }
  TEUCHOS_TEST_FOR_EXCEPTION (eval_on_side!=dl->isSideLayouts, std::logic_error,
      "Error! Input Layouts structure not compatible with requested field layout.\n");

  Teuchos::RCP<PHX::DataLayout> layout;
  if (p.isParameter("Nodal") && p.get<bool>("Nodal")) {
    layout = eval_on_side ? dl->node_scalar_sideset : dl->node_scalar;
  } else {
    layout = eval_on_side ? dl->qp_scalar_sideset : dl->qp_scalar;
  }

  numPts = layout->extent(1);

  P_w   = decltype(P_w)(p.get<std::string> ("Water Pressure Variable Name"), layout);
  phi_0 = decltype(phi_0)(p.get<std::string> ("Basal Gravitational Water Potential Variable Name"), layout);
  phi   = decltype(phi)(p.get<std::string> ("Hydraulic Potential Variable Name"), layout);

  this->addDependentField (P_w);
  this->addDependentField (phi_0);

  this->addEvaluatedField (phi);

  use_h = false;
  Teuchos::ParameterList& hydro_params = *p.get<Teuchos::ParameterList*>("LandIce Hydrology");
  if (hydro_params.get<bool>("Use Water Thickness In Effective Pressure Formula",false)) {
    use_h = true;

    h = decltype(h)(p.get<std::string> ("Water Thickness Variable Name"), layout);
    this->addDependentField(h);

    // Setting parameters
    Teuchos::ParameterList& physics  = *p.get<Teuchos::ParameterList*>("LandIce Physical Parameters");

    rho_w = physics.get<double>("Water Density",1000);
    g     = physics.get<double>("Gravity Acceleration",9.8);
  }

  this->setName("HydraulicPotential"+PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydraulicPotential<EvalT, Traits>::
evaluateFields (typename Traits::EvalData workset)
{
  if (eval_on_side) {
    evaluateFieldsSide(workset);
  } else {
    evaluateFieldsCell(workset);
  }
}

template<typename EvalT, typename Traits>
void HydraulicPotential<EvalT, Traits>::
evaluateFieldsSide (typename Traits::EvalData workset)
{
  if (workset.sideSets->find(sideSetName)==workset.sideSets->end()) return;

  ScalarT zero(0.0);
  sideSet = workset.sideSetViews->at(sideSetName);
  for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
  {
    for (unsigned int pt=0; pt<numPts; ++pt) {
      // Recall that phi is in kPa, but h is in m. Need to convert to km.
      phi(sideSet_idx,pt) = P_w(sideSet_idx,pt) + phi_0(sideSet_idx,pt) + (use_h ? rho_w*g*h(sideSet_idx,pt)/1000 : zero);
    }
  }
}

//**********************************************************************
template<typename EvalT, typename Traits>
void HydraulicPotential<EvalT, Traits>::
evaluateFieldsCell (typename Traits::EvalData workset)
{
  ScalarT zero(0.0);
  for (unsigned int cell=0; cell<workset.numCells; ++cell) {
    for (unsigned int pt=0; pt<numPts; ++pt) {
      // Recall that phi is in kPa, but h is in m. Need to convert to km.
      phi(cell,pt) = P_w(cell,pt) + phi_0(cell,pt) + (use_h ? rho_w*g*h(cell,pt)/1000 : zero);
    }
  }
}

} // Namespace LandIce
