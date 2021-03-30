//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_GatherBlockedSolution.hpp"

#include "Albany_BlockedSTKDiscretization.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_Macros.hpp"

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

#include <Kokkos_DynRankView.hpp>
#include <Phalanx_MDField.hpp>
#include <Teuchos_ParameterListExceptions.hpp>
#include <vector>
#include <string>
#include <chrono>

namespace PHAL {

template<typename EvalT, typename Traits>
GatherBlockedSolutionBase<EvalT,Traits>::
GatherBlockedSolutionBase(const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl)
{
  // Figure out if we need time derivativves
  if (p.isType<bool>("Disable Transient")) {
    enableTransient = !p.get<bool>("Disable Transient");
  } else {
    enableTransient = true;
  }

  if (p.isType<bool>("Enable Acceleration")) {
    enableAcceleration = p.get<bool>("Enable Acceleration");
  } else {
    enableAcceleration = false;
  }

  // Get solution names, and resize PHX fields arrays
  Teuchos::ArrayRCP<std::string> dof_names, dof_dot_names, dof_dotdot_names;

  dof_names = p.get< Teuchos::ArrayRCP<std::string> >("Solution Names");
  md_field.resize(num_blocks);
  num_blocks = dof_names.size();
  val.resize(num_blocks);

  if (enableTransient) {
    dof_dot_names = p.get< Teuchos::ArrayRCP<std::string> >("Solution Dot Names");
    TEUCHOS_TEST_FOR_EXCEPTION (num_blocks==dof_dot_names.size(), Teuchos::Exceptions::InvalidParameter,
      "Error! 'Solution Dot Names' array size should match that of 'Solution Names' array.\n");
    md_field_dot.resize(num_blocks);
    val_dot.resize(num_blocks);
  }

  if (enableAcceleration) {
    dof_dotdot_names = p.get< Teuchos::ArrayRCP<std::string> >("Solution Dot Dot Names");
    TEUCHOS_TEST_FOR_EXCEPTION (num_blocks==dof_dotdot_names.size(), Teuchos::Exceptions::InvalidParameter,
      "Error! 'Solution Dot Dot Names' array size should match that of 'Solution Names' array.\n");
    md_field_dotdot.resize(num_blocks);
    val_dotdot.resize(num_blocks);
  }

  // Get blocks specs (rank, size, index)
  std::vector<int> ranks, num_local_dofs, indices;
  if (p.isType<std::vector<int>>("Block Ranks")) {
    ranks = p.get<std::vector<int>>("Block Ranks");
  } else {
    ranks.resize(dof_names.size(),0);
  }
  TEUCHOS_TEST_FOR_EXCEPTION (num_blocks!=block_ranks.size(), Teuchos::Exceptions::InvalidParameter,
      "Error! 'Block Ranks' array size should match that of 'Solution Names' array.\n");

  if (p.isType<std::vector<int>>("Num Block Local Dofs")) {
    num_local_dofs = p.get<std::vector<int>>("Num Block Local Dofs");
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(!p.isType<Albany::BlockedSTKDiscretization>("Blocked Discretization"),
        Teuchos::Exceptions::InvalidParameter,
        "Error! Either provide the number of dofs for each block (as a std::vector<int>),\n"
        "       or the block discretization associated with this block solution.\n");
    num_local_dofs.resize(dof_names.size(),0);
  }
  TEUCHOS_TEST_FOR_EXCEPTION (num_blocks!=num_block_local_dofs.size(),
      Teuchos::Exceptions::InvalidParameter,
      "Error! 'Num Block Local Dofs' array size should match that of 'Solution Names' array.\n");

  if (p.isType<std::vector<int>>("Block Indices")) {
    indices = p.get<std::vector<int>>("Block Indices");
  } else {
    indices.resize(dof_names.size(),0);
    for (int i=0; i<num_blocks; ++i) {
      indices[i] = i;
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION (num_blocks!=block_indices.size(), Teuchos::Exceptions::InvalidParameter,
      "Error! 'Block Indices' array size should match that of 'Solution Names' array.\n");

  // Copy block info to device views
  block_indices.resize(num_blocks);
  num_block_local_dofs.resize(num_blocks);
  block_ranks.resize(num_blocks);
  for (int i=0; i<num_blocks; ++i) {
    block_indices.view_host()[i] = indices[i];
    num_block_local_dofs.view_host()[i] = num_local_dofs[i];
    block_ranks.view_host()[i] = ranks[i];
  }
  block_ranks.host_to_device();
  num_block_local_dofs.host_to_device();
  block_indices.host_to_device();

  nodeID.resize(num_blocks);

  // Create PHX fields
  for (int i=0; i<num_blocks; ++i) {
    using std::to_string;
    switch(block_ranks[i]) {
      case Scalar:
        md_field[i] = md_field_type(dof_names[i],dl->node_scalar);
        if (enableTransient) {
          md_field_dot[i] = md_field_type(dof_dot_names[i],dl->node_scalar);
        }
        if (enableAcceleration) {
          md_field_dotdot[i] = md_field_type(dof_dotdot_names[i],dl->node_scalar);
        }
        break;
      case Vector:
        md_field[i] = md_field_type(dof_names[i],dl->node_vector);
        if (enableTransient) {
          md_field_dot[i] = md_field_type(dof_dot_names[i],dl->node_vector);
        }
        if (enableAcceleration) {
          md_field_dotdot[i] = md_field_type(dof_dotdot_names[i],dl->node_vector);
        }
      case Tensor:
        md_field[i] = md_field_type(dof_names[i],dl->node_tensor);
        if (enableTransient) {
          md_field_dot[i] = md_field_type(dof_dot_names[i],dl->node_tensor);
        }
        if (enableAcceleration) {
          md_field_dotdot[i] = md_field_type(dof_dotdot_names[i],dl->node_tensor);
        }
      default:
        TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter,
            "Error! Unexpected rank (" + to_string + ") for block " + to_string(block_indices.view_host()[i]) + ".\n");
    }
    this->addEvaluatedField(md_field[i]);
    val[i] = md_field[i].get_static_view();

    if (enableTransient) {
      this->addEvaluatedField(md_field_dot[i]);
      val_dot[i] = md_field_dot[i].get_static_view();
    }
    if (enableAcceleration) {
      this->addEvaluatedField(md_field_dotdot[i]);
      val_dotdot[i] = md_field_dotdot[i].get_static_view();
    }
  }
  val.host_to_device();
  if (enableTransient) {
    val_dot.host_to_device();
  }
  if (enableAcceleration) {
    val_dotdot.host_to_device();
  }

  this->setName("Gather Solution"+PHX::print<EvalT>() );
}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherBlockedSolutionBase<EvalT,Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& /* fm */)
{
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields(),false);
}

// **********************************************************************

// **********************************************************************
// Specialization: Residual
// **********************************************************************

template<typename Traits>
GatherBlockedSolution<PHAL::AlbanyTraits::Residual, Traits>::
GatherBlockedSolution(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl)
  : GatherBlockedSolutionBase<PHAL::AlbanyTraits::Residual, Traits>(p,dl)
{
  // Nothing to do here
}

template<typename Traits>
GatherBlockedSolution<PHAL::AlbanyTraits::Residual, Traits>::
GatherBlockedSolution(const Teuchos::ParameterList& p)
  : GatherBlockedSolutionBase<PHAL::AlbanyTraits::Residual, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct"))
{
  // Nothing to do here
}

// ********************************************************************
// Kokkos functors for Residual
template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const PHAL_GatherSolRank1_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
      (this->valVec)(cell,node,eq)= x_constView(nodeID(cell, node,this->offset+eq));
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const PHAL_GatherSolRank1_Transient_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
      (this->valVec_dot)(cell,node,eq)= xdot_constView(nodeID(cell, node, this->offset+eq));
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const PHAL_GatherSolRank1_Acceleration_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
      (this->valVec_dotdot)(cell,node,eq)= xdotdot_constView(nodeID(cell, node, this->offset+eq));
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const PHAL_GatherSolRank2_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
      (this->valTensor)(cell,node,eq/numDim,eq%numDim)= x_constView(nodeID(cell, node, this->offset+eq));
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const PHAL_GatherSolRank2_Transient_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
      (this->valTensor_dot)(cell,node,eq/numDim,eq%numDim)= xdot_constView(nodeID(cell, node, this->offset+eq));
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const PHAL_GatherSolRank2_Acceleration_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
      (this->valTensor_dotdot)(cell,node,eq/numDim,eq%numDim)= xdotdot_constView(nodeID(cell, node, this->offset+eq));
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const PHAL_GatherSolRank0_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
      d_val[eq](cell,node)= x_constView(nodeID(cell, node, this->offset+eq));
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const PHAL_GatherSolRank0_Transient_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
      d_val_dot[eq](cell,node)= xdot_constView(nodeID(cell, node, this->offset+eq));
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const PHAL_GatherSolRank0_Acceleration_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
      d_val_dotdot[eq](cell,node)= xdotdot_constView(nodeID(cell, node, this->offset+eq));
}

// **********************************************************************
template<typename Traits>
void GatherBlockedSolution<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  const auto& x       = workset.x;
  const auto& xdot    = workset.xdot;
  const auto& xdotdot = workset.xdotdot;

#ifdef ALBANY_TIMER
  auto start = std::chrono::high_resolution_clock::now();
#endif

  // Get nodeIDs of each block on current workset
  for (int i=0; i<this->num_blocks; ++i) {
    nodeID.view_host()(i) = workset.blockedWsElNodeEqID[this->block_indices.view_host[i]];
  }
  nodeID.host_to_device();

  // Get device view of solution vector(s)
  x_constView = Albany::getDeviceData(x);
  if(!xdot.is_null()) {
    xdot_constView = Albany::getDeviceData(xdot);
  }
  if(!xdotdot.is_null()) {
    xdotdot_constView = Albany::getDeviceData(xdotdot);
  }

  if (this->tensorRank == 2){
    numDim = this->valTensor.extent(2);
    Kokkos::parallel_for(PHAL_GatherSolRank2_Policy(0,workset.numCells),*this);
    cudaCheckError();

    if (workset.transientTerms && this->enableTransient) {
      Kokkos::parallel_for(PHAL_GatherSolRank2_Transient_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }

    if (workset.accelerationTerms && this->enableAcceleration) {
      Kokkos::parallel_for(PHAL_GatherSolRank2_Acceleration_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }
  }

  else if (this->tensorRank == 1){
    Kokkos::parallel_for(PHAL_GatherSolRank1_Policy(0,workset.numCells),*this);
    cudaCheckError();

    if (workset.transientTerms && this->enableTransient) {
      Kokkos::parallel_for(PHAL_GatherSolRank1_Transient_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }

    if (workset.accelerationTerms && this->enableAcceleration) {
      Kokkos::parallel_for(PHAL_GatherSolRank1_Acceleration_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }
  }

  else {
    // Get MDField views from std::vector
    for (int i =0; i<numFields;i++){
      //val_kokkos[i]=this->val[i].get_view();
      val_kokkos[i]=this->val[i].get_static_view();
    }
    d_val=val_kokkos.template view<ExecutionSpace>();

    Kokkos::parallel_for(PHAL_GatherSolRank0_Policy(0,workset.numCells),*this);
    cudaCheckError();

    if (workset.transientTerms && this->enableTransient){
      // Get MDField views from std::vector
      for (int i =0; i<numFields;i++){
        //val_dot_kokkos[i]=this->val_dot[i].get_view();
        val_dot_kokkos[i]=this->val_dot[i].get_static_view();
      }
      d_val_dot=val_dot_kokkos.template view<ExecutionSpace>();

      Kokkos::parallel_for(PHAL_GatherSolRank0_Transient_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }
    if (workset.accelerationTerms && this->enableAcceleration){
      // Get MDField views from std::vector
      for (int i =0; i<numFields;i++){
        //val_dotdot_kokkos[i]=this->val_dotdot[i].get_view();
        val_dotdot_kokkos[i]=this->val_dotdot[i].get_static_view();
      }
      d_val_dotdot=val_dotdot_kokkos.template view<ExecutionSpace>();

      Kokkos::parallel_for(PHAL_GatherSolRank0_Acceleration_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }
  }

#ifdef ALBANY_TIMER
  PHX::Device::fence();
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  long long millisec= std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  std::cout<< "GaTher Solution Residual time = "  << millisec << "  "  << microseconds << std::endl;
#endif
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<typename Traits>
GatherBlockedSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
GatherBlockedSolution(const Teuchos::ParameterList& p,
          const Teuchos::RCP<Albany::Layouts>& dl) :
GatherBlockedSolutionBase<PHAL::AlbanyTraits::Jacobian, Traits>(p,dl),
numFields(GatherBlockedSolutionBase<PHAL::AlbanyTraits::Jacobian,Traits>::numFieldsBase)
{
}

template<typename Traits>
GatherBlockedSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
GatherBlockedSolution(const Teuchos::ParameterList& p) :
GatherBlockedSolutionBase<PHAL::AlbanyTraits::Jacobian, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct")),
numFields(GatherBlockedSolutionBase<PHAL::AlbanyTraits::Jacobian,Traits>::numFieldsBase)
{
}

//********************************************************************
////Kokkos functors for Jacobian
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJacRank2_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = (this->valTensor)(cell,node,eq/numDim,eq%numDim);
      valref=FadType(valref.size(), x_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) =j_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJacRank2_Transient_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = (this->valTensor_dot)(cell,node,eq/numDim,eq%numDim);
      valref =FadType(valref.size(), xdot_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) =m_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJacRank2_Acceleration_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = (this->valTensor_dotdot)(cell,node,eq/numDim,eq%numDim);
      valref=FadType(valref.size(), xdotdot_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) =n_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJacRank1_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; node++){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = (this->valVec)(cell,node,eq);
      valref =FadType(valref.size(), x_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) =j_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJacRank1_Transient_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = (this->valVec_dot)(cell,node,eq);
      valref =FadType(valref.size(), xdot_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) =m_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJacRank1_Acceleration_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = (this->valVec_dotdot)(cell,node,eq);
      valref =FadType(valref.size(), xdotdot_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) =n_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJacRank0_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = d_val[eq](cell,node);
      valref =FadType(valref.size(), x_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) =j_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJacRank0_Transient_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = d_val_dot[eq](cell,node);
      valref =FadType(valref.size(), xdot_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) =m_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJacRank0_Acceleration_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = d_val_dotdot[eq](cell,node);
      valref = FadType(valref.size(), xdotdot_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) = n_coeff;
    }
  }
}

#endif

// **********************************************************************
template<typename Traits>
void GatherBlockedSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  const auto& x       = workset.x;
  const auto& xdot    = workset.xdot;
  const auto& xdotdot = workset.xdotdot;

#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::ArrayRCP<const ST> x_constView, xdot_constView, xdotdot_constView;
  x_constView = Albany::getLocalData(x);
  if(!xdot.is_null()) {
    xdot_constView = Albany::getLocalData(xdot);
  }
  if(!xdotdot.is_null()) {
    xdot_constView = Albany::getLocalData(xdot);
  }

  int numDim = 0;
  if (this->tensorRank==2) numDim = this->valTensor.extent(2); // only needed for tensor fields

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const int neq = nodeID.extent(2);

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      int firstunk = neq * node + this->offset;
      for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = (this->tensorRank == 0 ? this->val[eq](cell,node) :
                    this->tensorRank == 1 ? this->valVec(cell,node,eq) :
                    this->valTensor(cell,node, eq/numDim, eq%numDim));
        valref = FadType(valref.size(), x_constView[nodeID(cell,node,this->offset + eq)]);
        valref.fastAccessDx(firstunk + eq) = workset.j_coeff;
      }
      if (workset.transientTerms && this->enableTransient) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = (this->tensorRank == 0 ? this->val_dot[eq](cell,node) :
                    this->tensorRank == 1 ? this->valVec_dot(cell,node,eq) :
                    this->valTensor_dot(cell,node, eq/numDim, eq%numDim));
        valref = FadType(valref.size(), xdot_constView[nodeID(cell,node,this->offset + eq)]);
        valref.fastAccessDx(firstunk + eq) = workset.m_coeff;
        }
      }
      if (workset.accelerationTerms && this->enableAcceleration) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = (this->tensorRank == 0 ? this->val_dotdot[eq](cell,node) :
                    this->tensorRank == 1 ? this->valVec_dotdot(cell,node,eq) :
                    this->valTensor_dotdot(cell,node, eq/numDim, eq%numDim));
        valref = FadType(valref.size(), xdotdot_constView[nodeID(cell,node,this->offset + eq)]);
        valref.fastAccessDx(firstunk + eq) = workset.n_coeff;
        }
      }
    }
  }

#else
#ifdef ALBANY_TIMER
  auto start = std::chrono::high_resolution_clock::now();
#endif

  // Get map for local data structures
  nodeID = workset.wsElNodeEqID;

  // Get dimensions and coefficients
  neq = nodeID.extent(2);
  j_coeff=workset.j_coeff;
  m_coeff=workset.m_coeff;
  n_coeff=workset.n_coeff;

  // Get vector view from a specific device
  x_constView = Albany::getDeviceData(x);
  if(!xdot.is_null()) {
    xdot_constView = Albany::getDeviceData(xdot);
  }
  if(!xdotdot.is_null()) {
    xdotdot_constView = Albany::getDeviceData(xdotdot);
  }

  if (this->tensorRank == 2) {
    numDim = this->valTensor.extent(2);

    Kokkos::parallel_for(PHAL_GatherJacRank2_Policy(0,workset.numCells),*this);
    cudaCheckError();

    if (workset.transientTerms && this->enableTransient) {
      Kokkos::parallel_for(PHAL_GatherJacRank2_Transient_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }

    if (workset.accelerationTerms && this->enableAcceleration) {
      Kokkos::parallel_for(PHAL_GatherJacRank2_Acceleration_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }
  }

  else if (this->tensorRank == 1) {
    Kokkos::parallel_for(PHAL_GatherJacRank1_Policy(0,workset.numCells),*this);
    cudaCheckError();

    if (workset.transientTerms && this->enableTransient) {
      Kokkos::parallel_for(PHAL_GatherJacRank1_Transient_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }

    if (workset.accelerationTerms && this->enableAcceleration) {
      Kokkos::parallel_for(PHAL_GatherJacRank1_Acceleration_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }
  }

  else {
    // Get MDField views from std::vector
    for (int i =0; i<numFields;i++) {
      //val_kokkos[i]=this->val[i].get_view();
      val_kokkos[i]=this->val[i].get_static_view();
    }
    d_val=val_kokkos.template view<ExecutionSpace>();

    Kokkos::parallel_for(PHAL_GatherJacRank0_Policy(0,workset.numCells),*this);
    cudaCheckError();

    if (workset.transientTerms && this->enableTransient) {
    // Get MDField views from std::vector
      for (int i =0; i<numFields;i++) {
        //val_dot_kokkos[i]=this->val_dot[i].get_view();
        val_dot_kokkos[i]=this->val_dot[i].get_static_view();
      }
      d_val_dot=val_dot_kokkos.template view<ExecutionSpace>();

      Kokkos::parallel_for(PHAL_GatherJacRank0_Transient_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }

    if (workset.accelerationTerms && this->enableAcceleration) {
    // Get MDField views from std::vector
      for (int i =0; i<numFields;i++) {
        //val_dotdot_kokkos[i]=this->val_dotdot[i].get_view();
        val_dotdot_kokkos[i]=this->val_dotdot[i].get_static_view();
      }
      d_val_dot=val_dotdot_kokkos.template view<ExecutionSpace>();

      Kokkos::parallel_for(PHAL_GatherJacRank0_Acceleration_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }
  }

#ifdef ALBANY_TIMER
  PHX::Device::fence();
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  long long millisec= std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  std::cout<< "GaTher Solution Jacobian time = "  << millisec << "  "  << microseconds << std::endl;
#endif
#endif
}

// **********************************************************************

// **********************************************************************
// Specialization: Tangent
// **********************************************************************

template<typename Traits>
GatherBlockedSolution<PHAL::AlbanyTraits::Tangent, Traits>::
GatherBlockedSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherBlockedSolutionBase<PHAL::AlbanyTraits::Tangent, Traits>(p,dl),
  numFields(GatherBlockedSolutionBase<PHAL::AlbanyTraits::Tangent,Traits>::numFieldsBase)
{
}

template<typename Traits>
GatherBlockedSolution<PHAL::AlbanyTraits::Tangent, Traits>::
GatherBlockedSolution(const Teuchos::ParameterList& p) :
  GatherBlockedSolutionBase<PHAL::AlbanyTraits::Tangent, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct")),
  numFields(GatherBlockedSolutionBase<PHAL::AlbanyTraits::Tangent,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void GatherBlockedSolution<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;

  const auto& x       = workset.x;
  const auto& xdot    = workset.xdot;
  const auto& xdotdot = workset.xdotdot;

  const auto& Vx       = workset.Vx;
  const auto& Vxdot    = workset.Vxdot;
  const auto& Vxdotdot = workset.Vxdotdot;

  //get const (read-only) view of x
  const auto& x_constView   = Albany::getLocalData(x);
  const auto& Vx_data       = Albany::getLocalData(Vx);
  const auto& Vxdot_data    = Albany::getLocalData(Vxdot);
  const auto& Vxdotdot_data = Albany::getLocalData(Vxdotdot);

  Teuchos::RCP<ParamVec> params = workset.params;
  //int num_cols_tot = workset.param_offset + workset.num_cols_p;

  int numDim = 0;
  if(this->tensorRank==2) {
    numDim = this->valTensor.extent(2); // only needed for tensor fields
  }

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = ((this->tensorRank == 2) ? (this->valTensor)(cell,node,eq/numDim,eq%numDim) :
                    (this->tensorRank == 1) ? (this->valVec)(cell,node,eq) :
                    (this->val[eq])(cell,node));
        if (Vx != Teuchos::null && workset.j_coeff != 0.0) {
          valref = TanFadType(valref.size(), x_constView[nodeID(cell,node,this->offset + eq)]);
          for (int k=0; k<workset.num_cols_x; k++)
            valref.fastAccessDx(k) =
              workset.j_coeff*Vx_data[k][nodeID(cell,node,this->offset + eq)];
        }
        else
          valref = TanFadType(x_constView[nodeID(cell,node,this->offset + eq)]);
      }
   }


   if (workset.transientTerms && this->enableTransient) {
    Teuchos::ArrayRCP<const ST> xdot_constView = Albany::getLocalData(xdot);
    for (std::size_t node = 0; node < this->numNodes; ++node) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = ((this->tensorRank == 2) ? (this->valTensor_dot)(cell,node,eq/numDim,eq%numDim) :
                    (this->tensorRank == 1) ? (this->valVec_dot)(cell,node,eq) :
                    (this->val_dot[eq])(cell,node));
          valref = TanFadType(valref.size(), xdot_constView[nodeID(cell,node,this->offset + eq)]);
          if (Vxdot != Teuchos::null && workset.m_coeff != 0.0) {
            for (int k=0; k<workset.num_cols_x; k++)
              valref.fastAccessDx(k) =
                workset.m_coeff*Vxdot_data[k][nodeID(cell,node,this->offset + eq)];
          }
        }
      }
   }

   if (workset.accelerationTerms && this->enableAcceleration) {
    Teuchos::ArrayRCP<const ST> xdotdot_constView = Albany::getLocalData(xdotdot);
    for (std::size_t node = 0; node < this->numNodes; ++node) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = ((this->tensorRank == 2) ? (this->valTensor_dotdot)(cell,node,eq/numDim,eq%numDim) :
                    (this->tensorRank == 1) ? (this->valVec_dotdot)(cell,node,eq) :
                    (this->val_dotdot[eq])(cell,node));

          valref = TanFadType(valref.size(), xdotdot_constView[nodeID(cell,node,this->offset + eq)]);
          if (Vxdotdot != Teuchos::null && workset.n_coeff != 0.0) {
            for (int k=0; k<workset.num_cols_x; k++)
              valref.fastAccessDx(k) =
                workset.n_coeff*Vxdotdot_data[k][nodeID(cell,node,this->offset + eq)];
          }
        }
      }
    }
  }
}

// **********************************************************************

// **********************************************************************
// Specialization: Distributed Parameter Derivative
// **********************************************************************

template<typename Traits>
GatherBlockedSolution<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
GatherBlockedSolution(const Teuchos::ParameterList& p,
               const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherBlockedSolutionBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p,dl),
  numFields(GatherBlockedSolutionBase<PHAL::AlbanyTraits::DistParamDeriv,Traits>::numFieldsBase)
{
}

template<typename Traits>
GatherBlockedSolution<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
GatherBlockedSolution(const Teuchos::ParameterList& p) :
  GatherBlockedSolutionBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct")),
  numFields(GatherBlockedSolutionBase<PHAL::AlbanyTraits::DistParamDeriv,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void GatherBlockedSolution<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;
  const auto& x       = workset.x;
  const auto& xdot    = workset.xdot;
  const auto& xdotdot = workset.xdotdot;

  //get const (read-only) view of x and xdot
  const auto& x_constView = Albany::getLocalData(x);

  if (this->tensorRank == 1) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      for (std::size_t node = 0; node < this->numNodes; ++node) {
        for (std::size_t eq = 0; eq < numFields; eq++)
          (this->valVec)(cell,node,eq) = x_constView[nodeID(cell,node,this->offset + eq)];
      }

    if (workset.transientTerms && this->enableTransient) {
      Teuchos::ArrayRCP<const ST> xdot_constView = Albany::getLocalData(xdot);
      for (std::size_t node = 0; node < this->numNodes; ++node) {
          for (std::size_t eq = 0; eq < numFields; eq++)
            (this->valVec_dot)(cell,node,eq) = xdot_constView[nodeID(cell,node,this->offset + eq)];
      }
    }

    if (workset.accelerationTerms && this->enableAcceleration) {
      Teuchos::ArrayRCP<const ST> xdotdot_constView = Albany::getLocalData(xdotdot);
      for (std::size_t node = 0; node < this->numNodes; ++node) {
          for (std::size_t eq = 0; eq < numFields; eq++)
            (this->valVec_dotdot)(cell,node,eq) = xdotdot_constView[nodeID(cell,node,this->offset + eq)];
        }
      }
    }
  } else
  if (this->tensorRank == 2) {
    int numDim = this->valTensor.extent(2);
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      for (std::size_t node = 0; node < this->numNodes; ++node) {
        for (std::size_t eq = 0; eq < numFields; eq++)
          (this->valTensor)(cell,node,eq/numDim,eq%numDim) = x_constView[nodeID(cell,node,this->offset + eq)];
      }

    if (workset.transientTerms && this->enableTransient) {
      Teuchos::ArrayRCP<const ST> xdot_constView = Albany::getLocalData(xdot);
      for (std::size_t node = 0; node < this->numNodes; ++node) {
          for (std::size_t eq = 0; eq < numFields; eq++)
            (this->valTensor_dot)(cell,node,eq/numDim,eq%numDim) = xdot_constView[nodeID(cell,node,this->offset + eq)];
      }
    }

    if (workset.accelerationTerms && this->enableAcceleration) {
      Teuchos::ArrayRCP<const ST> xdotdot_constView = Albany::getLocalData(xdotdot);
      for (std::size_t node = 0; node < this->numNodes; ++node) {
          for (std::size_t eq = 0; eq < numFields; eq++)
            (this->valTensor_dotdot)(cell,node,eq/numDim,eq%numDim) = xdotdot_constView[nodeID(cell,node,this->offset + eq)];
        }
      }
    }
  } else {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      for (std::size_t node = 0; node < this->numNodes; ++node) {
        for (std::size_t eq = 0; eq < numFields; eq++)
          (this->val[eq])(cell,node) = x_constView[nodeID(cell,node,this->offset + eq)];
      }
    if (workset.transientTerms && this->enableTransient) {
      Teuchos::ArrayRCP<const ST> xdot_constView = Albany::getLocalData(xdot);
      for (std::size_t node = 0; node < this->numNodes; ++node) {
          for (std::size_t eq = 0; eq < numFields; eq++)
            (this->val_dot[eq])(cell,node) = xdot_constView[nodeID(cell,node,this->offset + eq)];
      }
    }

    if (workset.accelerationTerms && this->enableAcceleration) {
      Teuchos::ArrayRCP<const ST> xdotdot_constView = Albany::getLocalData(xdotdot);
      for (std::size_t node = 0; node < this->numNodes; ++node) {
          for (std::size_t eq = 0; eq < numFields; eq++)
            (this->val_dotdot[eq])(cell,node) = xdotdot_constView[nodeID(cell,node,this->offset + eq)];
        }
      }
    }
  }
}

// **********************************************************************
// Specialization: HessianVec
// **********************************************************************

template<typename Traits>
GatherBlockedSolution<PHAL::AlbanyTraits::HessianVec, Traits>::
GatherBlockedSolution(const Teuchos::ParameterList& p,
               const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherBlockedSolutionBase<PHAL::AlbanyTraits::HessianVec, Traits>(p,dl),
  numFields(GatherBlockedSolutionBase<PHAL::AlbanyTraits::HessianVec,Traits>::numFieldsBase)
{
}

template<typename Traits>
GatherBlockedSolution<PHAL::AlbanyTraits::HessianVec, Traits>::
GatherBlockedSolution(const Teuchos::ParameterList& p) :
  GatherBlockedSolutionBase<PHAL::AlbanyTraits::HessianVec, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct")),
  numFields(GatherBlockedSolutionBase<PHAL::AlbanyTraits::HessianVec,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void GatherBlockedSolution<PHAL::AlbanyTraits::HessianVec, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  const auto& x       = workset.x;
  const auto& xdot    = workset.xdot;
  const auto& xdotdot = workset.xdotdot;

  Teuchos::RCP<const Thyra_MultiVector> direction_x = workset.hessianWorkset.direction_x;

  auto nodeID = workset.wsElNodeEqID;
  Teuchos::ArrayRCP<const ST> x_constView, xdot_constView, xdotdot_constView, direction_x_constView;
  bool g_xx_is_active = !workset.hessianWorkset.hess_vec_prod_g_xx.is_null();
  bool g_xp_is_active = !workset.hessianWorkset.hess_vec_prod_g_xp.is_null();
  bool g_px_is_active = !workset.hessianWorkset.hess_vec_prod_g_px.is_null();
  bool f_xx_is_active = !workset.hessianWorkset.hess_vec_prod_f_xx.is_null();
  bool f_xp_is_active = !workset.hessianWorkset.hess_vec_prod_f_xp.is_null();
  bool f_px_is_active = !workset.hessianWorkset.hess_vec_prod_f_px.is_null();
  x_constView = Albany::getLocalData(x);
  if(!xdot.is_null()) {
    xdot_constView = Albany::getLocalData(xdot);
  }
  if(!xdotdot.is_null()) {
    xdot_constView = Albany::getLocalData(xdot);
  }

  // is_x_active is true if we compute the Hessian-vector product contributions of either:
  // Hv_g_xx, Hv_g_xp, Hv_f_xx, or Hv_f_xp, i.e. if the first derivative is w.r.t. the solution.
  // If one of those is active, we have to initialize the first level of AD derivatives:
  // .fastAccessDx().val().
  const bool is_x_active = g_xx_is_active || g_xp_is_active || f_xx_is_active || f_xp_is_active;

  // is_x_direction_active is true if we compute the Hessian-vector product contributions of either:
  // Hv_g_xx, Hv_g_px, Hv_f_xx, or Hv_f_px, i.e. if the second derivative is w.r.t. the solution direction.
  // If one of those is active, we have to initialize the second level of AD derivatives:
  // .val().fastAccessDx().
  const bool is_x_direction_active = g_xx_is_active || g_px_is_active || f_xx_is_active || f_px_is_active;

  if(is_x_direction_active) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        direction_x.is_null(),
        Teuchos::Exceptions::InvalidParameter,
        "\nError in GatherBlockedSolution<HessianVec, Traits>: "
        "direction_x is not set and the direction is active.\n");
    direction_x_constView = Albany::getLocalData(direction_x->col(0));
  }

  int numDim = 0;
  if (this->tensorRank==2) numDim = this->valTensor.extent(2); // only needed for tensor fields

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const int neq = nodeID.extent(2);

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      int firstunk = neq * node + this->offset;
      for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = (this->tensorRank == 0 ? this->val[eq](cell,node) :
                    this->tensorRank == 1 ? this->valVec(cell,node,eq) :
                    this->valTensor(cell,node, eq/numDim, eq%numDim));
        RealType xvec_val = x_constView[nodeID(cell,node,this->offset + eq)];

        valref = HessianVecFad(valref.size(), xvec_val);
        // If we differentiate w.r.t. the solution, we have to set the first
        // derivative to 1
        if (is_x_active)
          valref.fastAccessDx(firstunk + eq).val() = 1;
        // If we differentiate w.r.t. the solution direction, we have to set
        // the second derivative to the related direction value
        if (is_x_direction_active)
          valref.val().fastAccessDx(0) = direction_x_constView[nodeID(cell,node,this->offset + eq)];
      }
    }
  }

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      if (workset.transientTerms && this->enableTransient) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          typename PHAL::Ref<ScalarT>::type
            valref = (this->tensorRank == 0 ? this->val_dot[eq](cell,node) :
                      this->tensorRank == 1 ? this->valVec_dot(cell,node,eq) :
                      this->valTensor_dot(cell,node, eq/numDim, eq%numDim));
          valref = xdot_constView[nodeID(cell,node,this->offset + eq)];
        }
      }
      if (workset.accelerationTerms && this->enableAcceleration) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          typename PHAL::Ref<ScalarT>::type
            valref = (this->tensorRank == 0 ? this->val_dotdot[eq](cell,node) :
                      this->tensorRank == 1 ? this->valVec_dotdot(cell,node,eq) :
                      this->valTensor_dotdot(cell,node, eq/numDim, eq%numDim));
          valref = xdotdot_constView[nodeID(cell,node,this->offset + eq)];
        }
      }
    }
  }
}

} // namespace PHAL
