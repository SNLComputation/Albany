//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_GATHER_BLOCKED_SOLUTION_HPP
#define PHAL_GATHER_BLOCKED_SOLUTION_HPP

#include "Albany_KokkosTypes.hpp"
#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "Albany_DiscretizationUtils.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dimension.hpp"

#include "Teuchos_ParameterList.hpp"

#include "Kokkos_Vector.hpp"

namespace PHAL {
/** \brief Gathers solution values from the Newton solution vector into
    the nodal fields of the field manager

    Currently makes an assumption that the stride is constant for dofs
    and that the nmber of dofs is equal to the size of the solution
    names vector.

*/
// **************************************************************
// Base Class with Generic Implementations: Specializations for
// Automatic Differentiation Below
// **************************************************************

template<typename EvalT, typename Traits>
class GatherBlockedSolutionBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                           public PHX::EvaluatorDerived<EvalT, Traits>
{
public:
  GatherBlockedSolutionBase(const Teuchos::ParameterList& p,
                          const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);

  // This function requires template specialization, in derived class below
  virtual void evaluateFields(typename Traits::EvalData d) = 0;

protected:

  // Some typedefs
  using ScalarT = typename EvalT::ScalarT;
  using md_field_type = PHX::MDField<ScalarT>;

  template<typename T>
  using DynRankView = Kokkos::DynRankView<T, PHX::Device>;

  template<typename T>
  using View = Kokkos::View<T, PHX::Device>;

  template<typename T>
  using KV = Kokkos::vector<T>;

  // Flags to enable gathering of time derivatives
  bool enableTransient;
  bool enableAcceleration;

  // The gathered MD fields
  Teuchos::Array<md_field_type> md_field;
  Teuchos::Array<md_field_type> md_field_dot;
  Teuchos::Array<md_field_type> md_field_dotdot;

  // The number of blocks gathered by this evaluator
  int num_blocks;

  // Kokkos-related structures

  KV<Albany::WorksetConn> nodeID;
  KV<Albany::DeviceView1d<const ST>> x, x_dot, x_dotdot;
  KV<DynRankView<ScalarT>> val, val_dot, val_dotdot;

  // The index of each block (where to fetch the data from the Thyra_Product(Multi)Vector)
  KV<int> block_indices;

  // The rank of each block (0=scalar, 1=vector, 2=tensor)
  enum : int {
    Scalar = 0,
    Vector = 1,
    Tensor = 2
  };
  KV<int> block_ranks;

  // Number of local Dofs for each block. For Lagrangian FE, these are the mesh nodes,
  // but for other FE type they are not necessarily associated to a unique geometric entity.
  KV<int> num_block_local_dofs;
};

template<typename EvalT, typename Traits> class GatherBlockedSolution;

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************


// **************************************************************
// Residual
// **************************************************************
template<typename Traits>
class GatherBlockedSolution<PHAL::AlbanyTraits::Residual,Traits>
   : public GatherBlockedSolutionBase<PHAL::AlbanyTraits::Residual, Traits>  {

public:
  GatherBlockedSolution(const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl);

  // Old constructor, still needed by BCs that use PHX Factory
  GatherBlockedSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);

private:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  const int numFields;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:
  struct PHAL_GatherSolRank2_Tag{};
  struct PHAL_GatherSolRank2_Transient_Tag{};
  struct PHAL_GatherSolRank2_Acceleration_Tag{};

  struct PHAL_GatherSolRank1_Tag{};
  struct PHAL_GatherSolRank1_Transient_Tag{};
  struct PHAL_GatherSolRank1_Acceleration_Tag{};

  struct PHAL_GatherSolRank0_Tag{};
  struct PHAL_GatherSolRank0_Transient_Tag{};
  struct PHAL_GatherSolRank0_Acceleration_Tag{};

  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherSolRank2_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherSolRank2_Transient_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherSolRank2_Acceleration_Tag&, const int& cell) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherSolRank1_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherSolRank1_Transient_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherSolRank1_Acceleration_Tag&, const int& cell) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherSolRank0_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherSolRank0_Transient_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherSolRank0_Acceleration_Tag&, const int& cell) const;

private:
  int numDim;

  typedef GatherBlockedSolutionBase<PHAL::AlbanyTraits::Residual, Traits> Base;
  using Base::nodeID;
  using Base::x_constView;
  using Base::xdot_constView;
  using Base::xdotdot_constView;
  using Base::val_kokkos;
  using Base::val_dot_kokkos;
  using Base::val_dotdot_kokkos;
  using Base::d_val;
  using Base::d_val_dot;
  using Base::d_val_dotdot;

  typedef typename PHX::Device::execution_space ExecutionSpace;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherSolRank2_Tag> PHAL_GatherSolRank2_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherSolRank2_Transient_Tag> PHAL_GatherSolRank2_Transient_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherSolRank2_Acceleration_Tag> PHAL_GatherSolRank2_Acceleration_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherSolRank1_Tag> PHAL_GatherSolRank1_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherSolRank1_Transient_Tag> PHAL_GatherSolRank1_Transient_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherSolRank1_Acceleration_Tag> PHAL_GatherSolRank1_Acceleration_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherSolRank0_Tag> PHAL_GatherSolRank0_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherSolRank0_Transient_Tag> PHAL_GatherSolRank0_Transient_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherSolRank0_Acceleration_Tag> PHAL_GatherSolRank0_Acceleration_Policy;

#endif
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class GatherBlockedSolution<PHAL::AlbanyTraits::Jacobian,Traits>
   : public GatherBlockedSolutionBase<PHAL::AlbanyTraits::Jacobian, Traits>  {

public:
  GatherBlockedSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  GatherBlockedSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);

private:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  const int numFields;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:
  struct PHAL_GatherJacRank2_Tag{};
  struct PHAL_GatherJacRank2_Transient_Tag{};
  struct PHAL_GatherJacRank2_Acceleration_Tag{};

  struct PHAL_GatherJacRank1_Tag{};
  struct PHAL_GatherJacRank1_Transient_Tag{};
  struct PHAL_GatherJacRank1_Acceleration_Tag{};

  struct PHAL_GatherJacRank0_Tag{};
  struct PHAL_GatherJacRank0_Transient_Tag{};
  struct PHAL_GatherJacRank0_Acceleration_Tag{};

  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherJacRank2_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherJacRank2_Transient_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherJacRank2_Acceleration_Tag&, const int& cell) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherJacRank1_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherJacRank1_Transient_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherJacRank1_Acceleration_Tag&, const int& cell) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherJacRank0_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherJacRank0_Transient_Tag&, const int& cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const PHAL_GatherJacRank0_Acceleration_Tag&, const int& cell) const;

private:
  int neq, numDim;
  double j_coeff, n_coeff, m_coeff;

  typedef GatherBlockedSolutionBase<PHAL::AlbanyTraits::Jacobian, Traits> Base;
  using Base::nodeID;
  using Base::x_constView;
  using Base::xdot_constView;
  using Base::xdotdot_constView;
  using Base::val_kokkos;
  using Base::val_dot_kokkos;
  using Base::val_dotdot_kokkos;
  using Base::d_val;
  using Base::d_val_dot;
  using Base::d_val_dotdot;

  typedef typename PHX::Device::execution_space ExecutionSpace;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherJacRank2_Tag> PHAL_GatherJacRank2_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherJacRank2_Transient_Tag> PHAL_GatherJacRank2_Transient_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherJacRank2_Acceleration_Tag> PHAL_GatherJacRank2_Acceleration_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherJacRank1_Tag> PHAL_GatherJacRank1_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherJacRank1_Transient_Tag> PHAL_GatherJacRank1_Transient_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherJacRank1_Acceleration_Tag> PHAL_GatherJacRank1_Acceleration_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherJacRank0_Tag> PHAL_GatherJacRank0_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherJacRank0_Transient_Tag> PHAL_GatherJacRank0_Transient_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,PHAL_GatherJacRank0_Acceleration_Tag> PHAL_GatherJacRank0_Acceleration_Policy;

#endif
};


// **************************************************************
// Tangent (Jacobian mat-vec + parameter derivatives)
// **************************************************************
template<typename Traits>
class GatherBlockedSolution<PHAL::AlbanyTraits::Tangent,Traits>
   : public GatherBlockedSolutionBase<PHAL::AlbanyTraits::Tangent, Traits>  {

public:
  GatherBlockedSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  GatherBlockedSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
  typedef typename Kokkos::View<ScalarT*, PHX::Device>::reference_type reference_type;
  const std::size_t numFields;
};

// **************************************************************
// Distributed Parameter Derivative
// **************************************************************
template<typename Traits>
class GatherBlockedSolution<PHAL::AlbanyTraits::DistParamDeriv,Traits>
   : public GatherBlockedSolutionBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>  {

public:
  GatherBlockedSolution(const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl);
  GatherBlockedSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
  const std::size_t numFields;
};

// **************************************************************
// HessianVec
// **************************************************************

/**
 * @brief Template specialization of the GatherBlockedSolution Class for PHAL::AlbanyTraits::HessianVec EvaluationType.
 *
 * This specialization is used to gather the solution for the computation of:
 * <ul>
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}}\f$ contribution of the Hessian-vector product of the response function:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{x}} g(\boldsymbol{x}+ r\,\boldsymbol{v}_{\boldsymbol{x}},\boldsymbol{p}_1, \boldsymbol{p}_2)\right|_{r=0},
 *       \f]
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(g)\boldsymbol{v}_{\boldsymbol{p}_1} \f$ contribution of the Hessian-vector product of the response function:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(g)\boldsymbol{v}_{\boldsymbol{p}_1}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{x}} g(\boldsymbol{x}, \boldsymbol{p}_1+ r\,\boldsymbol{v}_{\boldsymbol{p}_1}, \boldsymbol{p}_2)\right|_{r=0},
 *       \f]
 *  <li> The \f$ \boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}} \f$ contribution of the Hessian-vector product of the response function:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{p}_2} g(\boldsymbol{x}+ r\,\boldsymbol{v}_{\boldsymbol{x}},\boldsymbol{p}_1, \boldsymbol{p}_2)\right|_{r=0},
 *       \f]
 *  <li> The \f$ \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}} \f$ contribution of the Hessian-vector product of the residual:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{x}} \left\langle \boldsymbol{f}(\boldsymbol{x}+ r\,\boldsymbol{v}_{\boldsymbol{x}},\boldsymbol{p}_1, \boldsymbol{p}_2),\boldsymbol{z}\right\rangle\right|_{r=0},
 *       \f]
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_1} \f$ contribution of the Hessian-vector product of the residual:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_1}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{x}} \left\langle \boldsymbol{f}(\boldsymbol{x}, \boldsymbol{p}_1+ r\,\boldsymbol{v}_{\boldsymbol{p}_1}, \boldsymbol{p}_2),\boldsymbol{z}\right\rangle\right|_{r=0},
 *       \f]
 *  <li> The \f$\boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}} \f$ contribution of the Hessian-vector product of the residual:
 *       \f[
 *         \boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}}=
 *         \left.\frac{\partial}{\partial r} \nabla_{\boldsymbol{p}_2} \left\langle \boldsymbol{f}(\boldsymbol{x}+ r\,\boldsymbol{v}_{\boldsymbol{x}},\boldsymbol{p}_1, \boldsymbol{p}_2),\boldsymbol{z}\right\rangle\right|_{r=0},
 *       \f]
 * </ul>
 *
 *  where \f$\boldsymbol{x}\f$ is the solution, \f$\boldsymbol{p}_1\f$ is a first parameter, \f$\boldsymbol{p}_2\f$ is a potentially different second parameter,
 * \f$g\f$ is the response function, \f$\boldsymbol{f}\f$ is the residual, \f$\boldsymbol{z}\f$ is the Lagrange multiplier vector, \f$\boldsymbol{v}_{\boldsymbol{x}}\f$ is a direction vector
 *  with the same dimension as the vector \f$\boldsymbol{x}\f$, and \f$\boldsymbol{v}_{\boldsymbol{p}_1} \f$ is a direction vector with the same dimension as the vector \f$\boldsymbol{p}_1\f$.
 * 
 * This gather is used when calling: 
 * <ul>
 *   <li> Albany::Application::evaluateResponse_HessVecProd_xx,
 *   <li> Albany::Application::evaluateResponse_HessVecProd_xp,
 *   <li> Albany::Application::evaluateResponse_HessVecProd_px, 
 *   <li> Albany::Application::evaluateResponse_HessVecProd_pp,
 *   <li> Albany::Application::evaluateResidual_HessVecProd_xx,
 *   <li> Albany::Application::evaluateResidual_HessVecProd_xp,
 *   <li> Albany::Application::evaluateResidual_HessVecProd_px, 
 *   <li> Albany::Application::evaluateResidual_HessVecProd_pp.
 * </ul>
 */

template<typename Traits>
class GatherBlockedSolution<PHAL::AlbanyTraits::HessianVec,Traits>
   : public GatherBlockedSolutionBase<PHAL::AlbanyTraits::HessianVec, Traits>  {

public:
  GatherBlockedSolution(const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl);
  GatherBlockedSolution(const Teuchos::ParameterList& p);

  /**
   * @brief Gather the solution for the Hessian-vector product computations.
   * 
   * The PHAL::AlbanyTraits::HessianVec::ScalarT is a nested Sacado::FAD type with two levels of
   * differentiation. 
   * 
   * This member function behaves in four different ways depending on which Hessian-vector product
   * contribution is currently computed:
   * 
   * <ol>
   * 
   * <li> If the contribution which is currently computed is one of the following contributions:
   * <ul>
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}}\f$,
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}}\f$,
   * </ul>
   * the first derivative and the second derivative are w.r.t the solution and the values of the FAD types have to be
   * initialized for both first and second derivatives.
   * 
   * Such a case is illustrated in the following table; the value of the solution, the first derivative, and the second derivative have to be set.
   * 
   * <table>
   * <caption id="multi_row">Example of the initialization of a solution</caption>
   * <tr><th class="background">     
   * <div><span class="bottom">First derivative</span>
   *   <span class="top">Second derivative</span>
   *   <div class="line"></div>
   * </div>                    <th class="cell">val        <th class="cell">dx(0)
   * <tr><th>val                  <td>65          <td>0.4
   * <tr><th>dx(0)                <td>1          <td>0
   * <tr><th>dx(1)                <td>0          <td>0
   * <tr><th>dx(2)                <td>0          <td>0
   * <tr><th>dx(3)                <td>0          <td>0
   * </table>
   * 
   * In the implementation, this is translated as follows:
   * <tt>
   *  is_x_active = true;
   *  is_x_direction_active = true;
   * </tt>
   * 
   * </li>
   * 
   * <li> If the contribution which is currently computed is one of the following contributions:
   * <ul>
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(g)\boldsymbol{v}_{\boldsymbol{p}_1}\f$,
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_1}\f$,
   * </ul>
   * only the first derivative is w.r.t the solution and the values of the FAD types have to be
   * initialized for the first derivatives.
   * 
   * Such a case is illustrated in the following table; the value of the solution and the first derivative have to be set.
   * 
   * <table>
   * <caption id="multi_row">Example of the initialization of a solution</caption>
   * <tr><th class="background">     
   * <div><span class="bottom">First derivative</span>
   *   <span class="top">Second derivative</span>
   *   <div class="line"></div>
   * </div>                    <th class="cell">val        <th class="cell">dx(0)
   * <tr><th>val                  <td>65          <td>0
   * <tr><th>dx(0)                <td>1          <td>0
   * <tr><th>dx(1)                <td>0          <td>0
   * <tr><th>dx(2)                <td>0          <td>0
   * <tr><th>dx(3)                <td>0          <td>0
   * </table>
   * 
   * In the implementation, this is translated as follows:
   * <tt>
   *  is_x_active = true;
   *  is_x_direction_active = false;
   * </tt>
   * 
   * </li>
   * 
   * <li> If the contribution which is currently computed is one of the following contributions:
   * <ul>
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}}\f$,
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}}\f$,
   * </ul>
   * only the second derivative is w.r.t the solution and the values of the FAD types have to be
   * initialized for the second derivatives.
   * 
   * Such a case is illustrated in the following table; the value of the solution and the second derivative have to be set.
   * 
   * <table>
   * <caption id="multi_row">Example of the initialization of a solution</caption>
   * <tr><th class="background">     
   * <div><span class="bottom">First derivative</span>
   *   <span class="top">Second derivative</span>
   *   <div class="line"></div>
   * </div>                    <th class="cell">val        <th class="cell">dx(0)
   * <tr><th>val                  <td>65          <td>0.4
   * <tr><th>dx(0)                <td>0          <td>0
   * <tr><th>dx(1)                <td>0          <td>0
   * <tr><th>dx(2)                <td>0          <td>0
   * <tr><th>dx(3)                <td>0          <td>0
   * </table>
   * 
   * In the implementation, this is translated as follows:
   * <tt>
   *  is_x_active = false;
   *  is_x_direction_active = true;
   * </tt>
   * 
   * </li>
   * 
   * <li> If the contribution which is currently computed is one of the following contributions:
   * <ul>
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{p}_1}(g)\boldsymbol{v}_{\boldsymbol{p}_1}\f$,
   *   <li> \f$\boldsymbol{H}_{\boldsymbol{p}_2\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_1}\f$,
   * </ul>
   * none of the derivative is w.r.t the solution.
   * The values of the derivatives of the FAD types should not be initialized during this function call.
   * 
   * Such a case is illustrated in the following table; only the value of the solution has to be set.
   * 
   * <table>
   * <caption id="multi_row">Example of the initialization of a solution</caption>
   * <tr><th class="background">     
   * <div><span class="bottom">First derivative</span>
   *   <span class="top">Second derivative</span>
   *   <div class="line"></div>
   * </div>                    <th class="cell">val        <th class="cell">dx(0)
   * <tr><th>val                  <td>65          <td>0
   * <tr><th>dx(0)                <td>0          <td>0
   * <tr><th>dx(1)                <td>0          <td>0
   * <tr><th>dx(2)                <td>0          <td>0
   * <tr><th>dx(3)                <td>0          <td>0
   * </table>
   * 
   * In the implementation, this is translated as follows:
   * <tt> is_x_active = false;
   *  is_x_direction_active = false;
   * </tt>
   * 
   * </li>
   * </ol>
   */
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::HessianVec::ScalarT ScalarT;
  const std::size_t numFields;
};

} // namespace PHAL

#endif // PHAL_GATHER_BLOCKED_SOLUTION_HPP
