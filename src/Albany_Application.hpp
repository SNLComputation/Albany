//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_APPLICATION_HPP
#define ALBANY_APPLICATION_HPP

#include "Albany_config.h"

#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_VerboseObject.hpp"

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_AbstractProblem.hpp"
#include "Albany_AbstractResponseFunction.hpp"
#include "Albany_StateManager.hpp"

#include "SolutionManager.hpp"
#include "Albany_DiscretizationFactory.hpp"
#include "Albany_BlockedSTKDiscretization.hpp"

#include "Sacado_ParameterAccessor.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Sacado_ScalarParameterLibrary.hpp"
#include "Sacado_ScalarParameterVector.hpp"

#include <Teuchos_RCPDecl.hpp>
#include <set>
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Setup.hpp"
#include "PHAL_Workset.hpp"

namespace Albany {

class Application
    : public Sacado::ParameterAccessor<PHAL::AlbanyTraits::Residual, SPL_Traits>
{
public:

  enum SolutionMethod
  {
    Steady,
    Transient,
    Continuation,
    Eigensolve
  };

  enum SolutionStatus
  {
    Converged,
    NotConverged
  } solutionStatus;

  //! Constructor(s) and Destructor
  Application (const Teuchos::RCP<const Teuchos_Comm>&     comm,
               const Teuchos::RCP<Teuchos::ParameterList>& params,
               const Teuchos::RCP<const Thyra_Vector>&     initial_guess = Teuchos::null);

  Application(const Teuchos::RCP<const Teuchos_Comm>& comm);

  Application(const Application&) = delete;

  ~Application() = default;

  //! Prohibit copying/moving
  Application& operator=(const Application&) = delete;
  Application& operator=(Application&&) = delete;

  void initialSetUp(const Teuchos::RCP<Teuchos::ParameterList>& params);
  void createMeshSpecs();
  void createMeshSpecs(Teuchos::RCP<Albany::AbstractMeshStruct> mesh);
  void buildProblem();
  void createDiscretization();
  void finalSetUp(const Teuchos::RCP<Teuchos::ParameterList>& params,
                  const Teuchos::RCP<const Thyra_Vector>& initial_guess = Teuchos::null);

  //! Get underlying abstract discretization
  Teuchos::RCP<Albany::AbstractDiscretization>
  getDiscretization() const;

  //! Get problem object
  Teuchos::RCP<Albany::AbstractProblem>
  getProblem() const;

  //! Get communicator
  Teuchos::RCP<const Teuchos_Comm>
  getComm() const;

  //! Get Thyra DOF vector space
  Teuchos::RCP<const Thyra_VectorSpace>
  getVectorSpace() const;

  //! Get Thyra DOF dual vector space
  Teuchos::RCP<const Thyra_VectorSpace>
  getDualVectorSpace() const;

  //! Create Jacobian operator
  Teuchos::RCP<Thyra_LinearOp>
  createJacobianOp() const;

  //! Get Preconditioner Operator
  Teuchos::RCP<Thyra_LinearOp>
  getPreconditioner();

  bool
  observeResponses() const
  {
    return observe_responses;
  }

  int
  observeResponsesFreq() const
  {
    return response_observ_freq;
  }

  Teuchos::Array<unsigned int>
  getMarkersForRelativeResponses() const
  {
    return relative_responses;
  }

  Teuchos::RCP<Albany::SolutionManager>
  getAdaptSolMgr()
  {
    return solMgr;
  }

  //! Get parameter library
  Teuchos::RCP<ParamLib>
  getParamLib() const;

  //! Get distributed parameter library
  Teuchos::RCP<DistributedParameterLibrary>
  getDistributedParameterLibrary() const;

  //! Get solution method
  SolutionMethod
  getSolutionMethod() const
  {
    return solMethod;
  }

  //! Get number of responses
  int
  getNumResponses() const;

  int
  getNumEquations() const
  {
    return neq;
  }
  int
  getSpatialDimension() const
  {
    return spatial_dimension;
  }
  int
  getTangentDerivDimension() const
  {
    return tangent_deriv_dim;
  }

  Teuchos::RCP<Albany::AbstractDiscretization>
  getDisc() const
  {
    return disc;
  }

  //! Get response function
  Teuchos::RCP<AbstractResponseFunction>
  getResponse(int i) const;


  SolutionStatus getSolutionStatus() const {
    return solutionStatus;
  }

  void setSolutionStatus(SolutionStatus status) {
    solutionStatus = status;
  }

  //! Return whether problem wants to use its own preconditioner
  bool
  suppliesPreconditioner() const;

  void
  computeGlobalResidual(
      const double                            current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& x_dot,
      const Teuchos::RCP<const Thyra_Vector>& x_dotdot,
      const Teuchos::Array<ParamVec>&         p,
      const Teuchos::RCP<Thyra_Vector>&       f,
      const double                            dt = 0.0);

 private:
  void
  computeGlobalResidualImpl(
      const double                           current_time,
      const Teuchos::RCP<const Thyra_Vector> x,
      const Teuchos::RCP<const Thyra_Vector> x_dot,
      const Teuchos::RCP<const Thyra_Vector> x_dotdot,
      const Teuchos::Array<ParamVec>&        p,
      const Teuchos::RCP<Thyra_Vector>&      f,
      const double                           dt = 0.0);

  PHAL::Workset
  set_dfm_workset(
      double const                            current_time,
      const Teuchos::RCP<const Thyra_Vector>  x,
      const Teuchos::RCP<const Thyra_Vector>  x_dot,
      const Teuchos::RCP<const Thyra_Vector>  x_dotdot,
      const Teuchos::RCP<Thyra_Vector>&       f,
      const Teuchos::RCP<const Thyra_Vector>& x_post_SDBCs = Teuchos::null);

 public:
  //! Compute global Jacobian
  /*!
   * Set xdot to NULL for steady-state problems
   */
  void
  computeGlobalJacobian(
      const double                            alpha,
      const double                            beta,
      const double                            omega,
      const double                            current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>&         p,
      const Teuchos::RCP<Thyra_Vector>&       f,
      const Teuchos::RCP<Thyra_LinearOp>&     jac,
      const double                            dt = 0.0);

 private:
  void
  computeGlobalJacobianImpl(
      const double                            alpha,
      const double                            beta,
      const double                            omega,
      const double                            current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>&         p,
      const Teuchos::RCP<Thyra_Vector>&       f,
      const Teuchos::RCP<Thyra_LinearOp>&     jac,
      const double                            dt = 0.0);

 public:
  //! Compute global Preconditioner
  /*!
   * Set xdot to NULL for steady-state problems
   */

  void
  computeGlobalTangent(
      const double                                 alpha,
      const double                                 beta,
      const double                                 omega,
      const double                                 current_time,
      bool                                         sum_derivs,
      const Teuchos::RCP<const Thyra_Vector>&      x,
      const Teuchos::RCP<const Thyra_Vector>&      xdot,
      const Teuchos::RCP<const Thyra_Vector>&      xdotdot,
      const Teuchos::Array<ParamVec>&              par,
      ParamVec*                                    deriv_par,
      const Teuchos::RCP<const Thyra_MultiVector>& Vx,
      const Teuchos::RCP<const Thyra_MultiVector>& Vxdot,
      const Teuchos::RCP<const Thyra_MultiVector>& Vxdotdot,
      const Teuchos::RCP<const Thyra_MultiVector>& Vp,
      const Teuchos::RCP<Thyra_Vector>&            f,
      const Teuchos::RCP<Thyra_MultiVector>&       JV,
      const Teuchos::RCP<Thyra_MultiVector>&       fp);

 public:
  //! Compute df/dp*V or (df/dp)^T*V for distributed parameter p
  /*!
   * Set xdot to NULL for steady-state problems
   */

  void
  applyGlobalDistParamDerivImpl(
      const double                                 current_time,
      const Teuchos::RCP<const Thyra_Vector>&      x,
      const Teuchos::RCP<const Thyra_Vector>&      xdot,
      const Teuchos::RCP<const Thyra_Vector>&      xdotdot,
      const Teuchos::Array<ParamVec>&              p,
      const std::string&                           dist_param_name,
      const bool                                   trans,
      const Teuchos::RCP<const Thyra_MultiVector>& V,
      const Teuchos::RCP<Thyra_MultiVector>&       fpV);

  //! Evaluate response functions
  /*!
   * Set xdot to NULL for steady-state problems
   */
  void
  evaluateResponse(
      int                                     response_index,
      const double                            current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>&         p,
      const Teuchos::RCP<Thyra_Vector>&       g);

  //! Evaluate tangent = alpha*dg/dx*Vx + beta*dg/dxdot*Vxdot + dg/dp*Vp
  /*!
   * Set xdot, dxdot_dp to NULL for steady-state problems
   */
  void
  evaluateResponseTangent(
      int                                          response_index,
      const double                                 alpha,
      const double                                 beta,
      const double                                 omega,
      const double                                 current_time,
      bool                                         sum_derivs,
      const Teuchos::RCP<const Thyra_Vector>&      x,
      const Teuchos::RCP<const Thyra_Vector>&      xdot,
      const Teuchos::RCP<const Thyra_Vector>&      xdotdot,
      const Teuchos::Array<ParamVec>&              p,
      ParamVec*                                    deriv_p,
      const Teuchos::RCP<const Thyra_MultiVector>& Vx,
      const Teuchos::RCP<const Thyra_MultiVector>& Vxdot,
      const Teuchos::RCP<const Thyra_MultiVector>& Vxdotdot,
      const Teuchos::RCP<const Thyra_MultiVector>& Vp,
      const Teuchos::RCP<Thyra_Vector>&            g,
      const Teuchos::RCP<Thyra_MultiVector>&       gx,
      const Teuchos::RCP<Thyra_MultiVector>&       gp);

  //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
  /*!
   * Set xdot, dg_dxdot to NULL for steady-state problems
   */
  void
  evaluateResponseDerivative(
      int                                     response_index,
      const double                            current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>&         p,
      ParamVec*                               deriv_p,
      const Teuchos::RCP<Thyra_Vector>&       g,
      const Thyra_Derivative&                 dg_dx,
      const Thyra_Derivative&                 dg_dxdot,
      const Thyra_Derivative&                 dg_dxdotdot,
      const Thyra_Derivative&                 dg_dp);

  void
  evaluateResponseDistParamDeriv(
      int                                     response_index,
      const double                            current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>&         param_array,
      const std::string&                      dist_param_name,
      const Teuchos::RCP<Thyra_MultiVector>&  dg_dp);

  /**
   * \brief evaluateResponse_HessVecProd_xx function
   * 
   * This function is used to compute contributions of the application of the Hessian \f$\boldsymbol{H}(g)\f$ of the <tt>response[response_index]</tt> 
   * (i.e. the response_index-th response) to a direction vector \f$\boldsymbol{v}\f$.
   * 
   * Representing the Hessian \f$\boldsymbol{H}(g)\f$ as a block matrix:
   * \f[
   *   \boldsymbol{H}(g) := 
   *   \begin{bmatrix} 
   *     \frac{\partial^{2} g}{\partial \boldsymbol{x}^{2}} & \frac{\partial^{2} g}{\partial \boldsymbol{x} \partial \boldsymbol{p}_{1}} & \cdots & \frac{\partial^{2} g}{\partial \boldsymbol{x} \partial \boldsymbol{p}_{n}} \\
   *     \frac{\partial^{2} g}{\partial \boldsymbol{p}_{1} \partial \boldsymbol{x}} & \frac{\partial^{2} g}{\partial \boldsymbol{p}_{1}^{2}} & \cdots & \frac{\partial^{2} g}{\partial \boldsymbol{p}_{1} \partial \boldsymbol{p}_{n}} \\
   *     \vdots & \vdots & \ddots & \vdots\\
   *     \frac{\partial^{2} g}{\partial \boldsymbol{p}_{n} \partial \boldsymbol{x}} & \frac{\partial^{2} g}{\partial \boldsymbol{p}_{n}\partial \boldsymbol{p}_{1}} & \cdots & \frac{\partial^{2} g}{\partial \boldsymbol{p}_{n}^{2}}
   *   \end{bmatrix} =
   *   \begin{bmatrix} 
   *     \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(g) & \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(g) & \cdots & \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_n}(g) \\ 
   *     \boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{x}}(g) & \boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{p}_1}(g) & \cdots & \boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{p}_n}(g) \\
   *     \vdots & \vdots & \ddots & \vdots\\
   *     \boldsymbol{H}_{\boldsymbol{p}_n\boldsymbol{x}}(g) & \boldsymbol{H}_{\boldsymbol{p}_n\boldsymbol{p}_1}(g) & \cdots & \boldsymbol{H}_{\boldsymbol{p}_n\boldsymbol{p}_n}(g) 
   *   \end{bmatrix},
   * \f]
   * where \f$g\f$ is the response, \f$\boldsymbol{x}\f$ is the solution vector, \f$\boldsymbol{p}_1\f$ is a first  parameter, \f$\ldots\f$, and \f$\boldsymbol{p}_n\f$ is the \f$n\f$-th 
   *  parameter, this function computes the contribution:
   * \f[
   *   \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}}.
   * \f]
   * 
   * This function can currently compute only one block contribution at a time; for one function call \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}}\f$
   * can be computed only for one fixed response index at a time.
   * 
   * \param response_index [in] Response index of the response which the Hessian is computed.
   * 
   * \param current_time [in] Current time at which the Hessian is computed for transient simulations.
   * 
   * \param v [in] Direction vector on which the Hessian is applied: \f$\boldsymbol{v}_{\boldsymbol{x}}\f$.
   * 
   * \param x [in] Solution vector for the current time step: \f$\boldsymbol{x}\f$.
   * 
   * \param xdot [in] Velocity vector for the current time step.
   * 
   * \param xdotdot [in] Acceleration vector for the current time step.
   * 
   * \param param_array [in] Array of the parameters vectors.
   * 
   * \param Hv_g_xx [out] the output of the computation: \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}}\f$.
   * 
   * This implementation relies on:
   * <ul>
   *  <li> PHAL::GatherSolution<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::GatherScalarNodalParameter<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::GatherScalarExtruded2DNodalParameter<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::SeparableScatterScalarResponse<PHAL::AlbanyTraits::HessianVec,Traits>.
   * </ul>
   */
  void
  evaluateResponse_HessVecProd_xx(
      int                                     response_index,
      const double                            current_time,
      const Teuchos::RCP<const Thyra_MultiVector>& v,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>&         param_array,
      const Teuchos::RCP<Thyra_MultiVector>&  Hv_g_xx);

  /**
   * \brief evaluateResponse_HessVecProd_xp function
   * 
   * This function is used to compute contributions of the application of the Hessian \f$\boldsymbol{H}(g)\f$ of the <tt>response[response_index]</tt> 
   * (i.e. the response_index-th response) to a direction vector \f$\boldsymbol{v}\f$.
   * 
   * Representing the Hessian \f$\boldsymbol{H}(g)\f$ as a block matrix:
   * \f[
   *   \boldsymbol{H}(g) := 
   *   \begin{bmatrix} 
   *     \frac{\partial^{2} g}{\partial \boldsymbol{x}^{2}} & \frac{\partial^{2} g}{\partial \boldsymbol{x} \partial \boldsymbol{p}_{1}} & \cdots & \frac{\partial^{2} g}{\partial \boldsymbol{x} \partial \boldsymbol{p}_{n}} \\
   *     \frac{\partial^{2} g}{\partial \boldsymbol{p}_{1} \partial \boldsymbol{x}} & \frac{\partial^{2} g}{\partial \boldsymbol{p}_{1}^{2}} & \cdots & \frac{\partial^{2} g}{\partial \boldsymbol{p}_{1} \partial \boldsymbol{p}_{n}} \\
   *     \vdots & \vdots & \ddots & \vdots\\
   *     \frac{\partial^{2} g}{\partial \boldsymbol{p}_{n} \partial \boldsymbol{x}} & \frac{\partial^{2} g}{\partial \boldsymbol{p}_{n}\partial \boldsymbol{p}_{1}} & \cdots & \frac{\partial^{2} g}{\partial \boldsymbol{p}_{n}^{2}}
   *   \end{bmatrix} =
   *   \begin{bmatrix} 
   *     \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(g) & \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(g) & \cdots & \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_n}(g) \\ 
   *     \boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{x}}(g) & \boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{p}_1}(g) & \cdots & \boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{p}_n}(g) \\
   *     \vdots & \vdots & \ddots & \vdots\\
   *     \boldsymbol{H}_{\boldsymbol{p}_n\boldsymbol{x}}(g) & \boldsymbol{H}_{\boldsymbol{p}_n\boldsymbol{p}_1}(g) & \cdots & \boldsymbol{H}_{\boldsymbol{p}_n\boldsymbol{p}_n}(g) 
   *   \end{bmatrix},
   * \f]
   * where \f$g\f$ is the response, \f$\boldsymbol{x}\f$ is the solution vector, \f$\boldsymbol{p}_1\f$ is a first  parameter, \f$\ldots\f$, and \f$\boldsymbol{p}_n\f$ is the \f$n\f$-th 
   *  parameter, this function computes the contribution:
   * \f[
   *   \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_j}(g)\boldsymbol{v}_{\boldsymbol{p}_j},
   * \f]
   * where \f$i\f$ and \f$j\f$ are 2 (potentially different) indices included in the range \f$\left[1,n\right]\f$.
   * 
   * This function can currently compute only one block contribution at a time; for one function call \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_j}(g)\boldsymbol{v}_{\boldsymbol{p}_j}\f$
   * can be computed only for one fixed \f$j\f$ and one fixed response index at a time.
   * 
   * \param response_index [in] Response index of the response which the Hessian is computed.
   * 
   * \param current_time [in] Current time at which the Hessian is computed for transient simulations.
   * 
   * \param v [in] Direction vector on which the Hessian is applied: \f$\boldsymbol{v}_{\boldsymbol{p}_j}\f$.
   * 
   * \param x [in] Solution vector for the current time step: \f$\boldsymbol{x}\f$.
   * 
   * \param xdot [in] Velocity vector for the current time step.
   * 
   * \param xdotdot [in] Acceleration vector for the current time step.
   * 
   * \param param_array [in] Array of the parameters vectors.
   * 
   * \param param_direction_name [in] Name of the parameter used for the second differentiation: \f$\boldsymbol{p}_j\f$.
   * 
   * \param Hv_g_xp [out] the output of the computation: \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_j}(g)\boldsymbol{v}_{\boldsymbol{p}_j}\f$.
   * 
   * This implementation relies on:
   * <ul>
   *  <li> PHAL::GatherSolution<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::GatherScalarNodalParameter<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::GatherScalarExtruded2DNodalParameter<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::SeparableScatterScalarResponse<PHAL::AlbanyTraits::HessianVec,Traits>.
   * </ul>
   */
  void
  evaluateResponse_HessVecProd_xp(
      int                                     response_index,
      const double                            current_time,
      const Teuchos::RCP<const Thyra_MultiVector>& v,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>&         param_array,
      const std::string&                      param_direction_name,
      const Teuchos::RCP<Thyra_MultiVector>&  Hv_g_xp);

  /**
   * \brief evaluateResponse_HessVecProd_px function
   * 
   * This function is used to compute contributions of the application of the Hessian \f$\boldsymbol{H}(g)\f$ of the <tt>response[response_index]</tt> 
   * (i.e. the response_index-th response) to a direction vector \f$\boldsymbol{v}\f$.
   * 
   * Representing the Hessian \f$\boldsymbol{H}(g)\f$ as a block matrix:
   * \f[
   *   \boldsymbol{H}(g) := 
   *   \begin{bmatrix} 
   *     \frac{\partial^{2} g}{\partial \boldsymbol{x}^{2}} & \frac{\partial^{2} g}{\partial \boldsymbol{x} \partial \boldsymbol{p}_{1}} & \cdots & \frac{\partial^{2} g}{\partial \boldsymbol{x} \partial \boldsymbol{p}_{n}} \\
   *     \frac{\partial^{2} g}{\partial \boldsymbol{p}_{1} \partial \boldsymbol{x}} & \frac{\partial^{2} g}{\partial \boldsymbol{p}_{1}^{2}} & \cdots & \frac{\partial^{2} g}{\partial \boldsymbol{p}_{1} \partial \boldsymbol{p}_{n}} \\
   *     \vdots & \vdots & \ddots & \vdots\\
   *     \frac{\partial^{2} g}{\partial \boldsymbol{p}_{n} \partial \boldsymbol{x}} & \frac{\partial^{2} g}{\partial \boldsymbol{p}_{n}\partial \boldsymbol{p}_{1}} & \cdots & \frac{\partial^{2} g}{\partial \boldsymbol{p}_{n}^{2}}
   *   \end{bmatrix} =
   *   \begin{bmatrix} 
   *     \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(g) & \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(g) & \cdots & \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_n}(g) \\ 
   *     \boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{x}}(g) & \boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{p}_1}(g) & \cdots & \boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{p}_n}(g) \\
   *     \vdots & \vdots & \ddots & \vdots\\
   *     \boldsymbol{H}_{\boldsymbol{p}_n\boldsymbol{x}}(g) & \boldsymbol{H}_{\boldsymbol{p}_n\boldsymbol{p}_1}(g) & \cdots & \boldsymbol{H}_{\boldsymbol{p}_n\boldsymbol{p}_n}(g) 
   *   \end{bmatrix},
   * \f]
   * where \f$g\f$ is the response, \f$\boldsymbol{x}\f$ is the solution vector, \f$\boldsymbol{p}_1\f$ is a first parameter, \f$\ldots\f$, and \f$\boldsymbol{p}_n\f$ is the \f$n\f$-th 
   * parameter, this function computes the contribution:
   * \f[
   *   \boldsymbol{H}_{\boldsymbol{p}_i\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}},
   * \f]
   * where \f$i\f$ is an index included in the range \f$\left[1,n\right]\f$.
   * 
   * This function can currently compute only one block contribution at a time; for one function call \f$\boldsymbol{H}_{\boldsymbol{p}_i\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}}\f$
   * can be computed only for one fixed \f$i\f$ and one fixed response index at a time.
   * 
   * \param response_index [in] Response index of the response which the Hessian is computed.
   * 
   * \param current_time [in] Current time at which the Hessian is computed for transient simulations.
   * 
   * \param v [in] Direction vector on which the Hessian is applied: \f$\boldsymbol{v}_{\boldsymbol{x}}\f$
   * 
   * \param x [in] Solution vector for the current time step: \f$\boldsymbol{x}\f$
   * 
   * \param xdot [in] Velocity vector for the current time step.
   * 
   * \param xdotdot [in] Acceleration vector for the current time step.
   * 
   * \param param_array [in] Array of the parameters vectors.
   * 
   * \param param_name [in] Name of the parameter used for the first differentiation: \f$\boldsymbol{p}_i\f$.
   * 
   * \param Hv_g_px [out] the output of the computation: \f$\boldsymbol{H}_{\boldsymbol{p}_i\boldsymbol{x}}(g)\boldsymbol{v}_{\boldsymbol{x}}\f$.
   * 
   * This implementation relies on:
   * <ul>
   *  <li> PHAL::GatherSolution<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::GatherScalarNodalParameter<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::GatherScalarExtruded2DNodalParameter<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::SeparableScatterScalarResponse<PHAL::AlbanyTraits::HessianVec,Traits>.
   * </ul>
   */
  void
  evaluateResponse_HessVecProd_px(
      int                                     response_index,
      const double                            current_time,
      const Teuchos::RCP<const Thyra_MultiVector>& v,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>&         param_array,
      const std::string&                      param_name,
      const Teuchos::RCP<Thyra_MultiVector>&  Hv_g_px);

  /**
   * \brief evaluateResponse_HessVecProd_pp function
   * 
   * This function is used to compute contributions of the application of the Hessian \f$\boldsymbol{H}(g)\f$ of the <tt>response[response_index]</tt> 
   * (i.e. the response_index-th response) to a direction vector \f$\boldsymbol{v}\f$.
   * 
   * Representing the Hessian \f$\boldsymbol{H}(g)\f$ as a block matrix:
   * \f[
   *   \boldsymbol{H}(g) := 
   *   \begin{bmatrix} 
   *     \frac{\partial^{2} g}{\partial \boldsymbol{x}^{2}} & \frac{\partial^{2} g}{\partial \boldsymbol{x} \partial \boldsymbol{p}_{1}} & \cdots & \frac{\partial^{2} g}{\partial \boldsymbol{x} \partial \boldsymbol{p}_{n}} \\
   *     \frac{\partial^{2} g}{\partial \boldsymbol{p}_{1} \partial \boldsymbol{x}} & \frac{\partial^{2} g}{\partial \boldsymbol{p}_{1}^{2}} & \cdots & \frac{\partial^{2} g}{\partial \boldsymbol{p}_{1} \partial \boldsymbol{p}_{n}} \\
   *     \vdots & \vdots & \ddots & \vdots\\
   *     \frac{\partial^{2} g}{\partial \boldsymbol{p}_{n} \partial \boldsymbol{x}} & \frac{\partial^{2} g}{\partial \boldsymbol{p}_{n}\partial \boldsymbol{p}_{1}} & \cdots & \frac{\partial^{2} g}{\partial \boldsymbol{p}_{n}^{2}}
   *   \end{bmatrix} =
   *   \begin{bmatrix} 
   *     \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(g) & \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(g) & \cdots & \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_n}(g) \\ 
   *     \boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{x}}(g) & \boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{p}_1}(g) & \cdots & \boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{p}_n}(g) \\
   *     \vdots & \vdots & \ddots & \vdots\\
   *     \boldsymbol{H}_{\boldsymbol{p}_n\boldsymbol{x}}(g) & \boldsymbol{H}_{\boldsymbol{p}_n\boldsymbol{p}_1}(g) & \cdots & \boldsymbol{H}_{\boldsymbol{p}_n\boldsymbol{p}_n}(g) 
   *   \end{bmatrix},
   * \f]
   * where \f$g\f$ is the response, \f$\boldsymbol{x}\f$ is the solution vector, \f$\boldsymbol{p}_1\f$ is a first  parameter, \f$\ldots\f$, and \f$\boldsymbol{p}_n\f$ is the \f$n\f$-th 
   *  parameter, this function computes the contribution:
   * \f[
   *   \boldsymbol{H}_{\boldsymbol{p}_i\boldsymbol{p}_j}(g)\boldsymbol{v}_{\boldsymbol{p}_j},
   * \f]
   * where \f$i\f$ and \f$j\f$ are 2 (potentially different) indices included in the range \f$\left[1,n\right]\f$.
   * 
   * This function can currently compute only one block contribution at a time; for one function call \f$\boldsymbol{H}_{\boldsymbol{p}_i\boldsymbol{p}_j}(g)\boldsymbol{v}_{\boldsymbol{p}_j}\f$
   * can be computed only for one fixed \f$i\f$, one fixed \f$j\f$, and one fixed response index at a time.
   * 
   * \param response_index [in] Response index of the response which the Hessian is computed.
   * 
   * \param current_time [in] Current time at which the Hessian is computed for transient simulations.
   * 
   * \param v [in] Direction vector on which the Hessian is applied: \f$\boldsymbol{v}_{\boldsymbol{p}_j}\f$.
   * 
   * \param x [in] Solution vector for the current time step: \f$\boldsymbol{x}\f$.
   * 
   * \param xdot [in] Velocity vector for the current time step.
   * 
   * \param xdotdot [in] Acceleration vector for the current time step.
   * 
   * \param param_array [in] Array of the parameters vectors.
   * 
   * \param param_name [in] Name of the parameter used for the first differentiation: \f$\boldsymbol{p}_i\f$.
   * 
   * \param param_direction_name [in] Name of the parameter used for the second differentiation: \f$\boldsymbol{p}_j\f$.
   * 
   * \param Hv_g_pp [out] the output of the computation: \f$\boldsymbol{H}_{\boldsymbol{p}_i\boldsymbol{p}_j}(g)\boldsymbol{v}_{\boldsymbol{p}_j}\f$.
   * 
   * This implementation relies on:
   * <ul>
   *  <li> PHAL::GatherSolution<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::GatherScalarNodalParameter<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::GatherScalarExtruded2DNodalParameter<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::SeparableScatterScalarResponse<PHAL::AlbanyTraits::HessianVec,Traits>.
   * </ul>
   */
  void
  evaluateResponse_HessVecProd_pp(
      int                                     response_index,
      const double                            current_time,
      const Teuchos::RCP<const Thyra_MultiVector>& v,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>&         param_array,
      const std::string&                      param_name,
      const std::string&                      param_direction_name,
      const Teuchos::RCP<Thyra_MultiVector>&  Hv_g_pp);

  /**
   * \brief evaluateResponseDistParamHessian_pp function
   *
   * This function is used to compute the Hessian \f$\boldsymbol{H}_{\boldsymbol{p}_i\boldsymbol{p}_i}(g)\f$ of the <tt>response[response_index]</tt> 
   * (i.e. the response_index-th response) where \f$g\f$ is the response and \f$\boldsymbol{p}_i\f$ is a distributed parameter.
   *
   * \param response_index [in] Response index of the response which the Hessian is computed.
   *
   * \param parameter_index [in] Parameter index of the distributed parameter.
   *
   * \param current_time [in] Current time at which the Hessian is computed for transient simulations.
   *
   * \param x [in] Solution vector for the current time step: \f$\boldsymbol{x}\f$.
   *
   * \param xdot [in] Velocity vector for the current time step.
   *
   * \param xdotdot [in] Acceleration vector for the current time step.
   *
   * \param param_array [in] Array of the parameters vectors.
   *
   * \param dist_param_name [in] Name of the distributed parameter.
   *
   * \param H [out] the output of the computation: \f$\boldsymbol{H}_{\boldsymbol{p}_i\boldsymbol{p}_i}(g)\f$.
   */
  void
  evaluateResponseDistParamHessian_pp(
      int                                     response_index,
      int                                     parameter_index,
      const double                            current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>&         param_array,
      const std::string&                      dist_param_name,
      const Teuchos::RCP<Thyra_LinearOp>&     H);

  /**
   * \brief evaluateResidual_HessVecProd_xx function
   * 
   * This function is used to compute contributions of the application of the Hessian \f$\boldsymbol{H}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\f$ of the inner product of the residual vector \f$\boldsymbol{f}\f$
   * and a Lagrange multiplier vector \f$\boldsymbol{z}\f$ to a direction vector \f$\boldsymbol{v}\f$.
   * 
   * Representing the Hessian \f$\boldsymbol{H}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\f$ as a block matrix:
   * \f[
   *   \boldsymbol{H}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) := 
   *   \begin{bmatrix} 
   *     \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{x}^{2}} & \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{x} \partial \boldsymbol{p}_{1}} & \cdots & \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{x} \partial \boldsymbol{p}_{n}} \\
   *     \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{p}_{1} \partial \boldsymbol{x}} & \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{p}_{1}^{2}} & \cdots & \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{p}_{1} \partial \boldsymbol{p}_{n}} \\
   *     \vdots & \vdots & \ddots & \vdots\\
   *     \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{p}_{n} \partial \boldsymbol{x}} & \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{p}_{n}\partial \boldsymbol{p}_{1}} & \cdots & \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{p}_{n}^{2}}
   *   \end{bmatrix} =
   *   \begin{bmatrix} 
   *     \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) & \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) & \cdots & \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_n}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) \\ 
   *     \boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) & \boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) & \cdots & \boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{p}_n}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) \\
   *     \vdots & \vdots & \ddots & \vdots\\
   *     \boldsymbol{H}_{\boldsymbol{p}_n\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) & \boldsymbol{H}_{\boldsymbol{p}_n\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) & \cdots & \boldsymbol{H}_{\boldsymbol{p}_n\boldsymbol{p}_n}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) 
   *   \end{bmatrix},
   * \f]
   * where \f$\boldsymbol{x}\f$ is the solution vector, \f$\boldsymbol{p}_1\f$ is a first  parameter, \f$\ldots\f$, and \f$\boldsymbol{p}_n\f$ is the \f$n\f$-th 
   *  parameter, this function computes the contribution:
   * \f[
   *   \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}}.
   * \f]
   * 
   * \param current_time [in] Current time at which the Hessian is computed for transient simulations.
   * 
   * \param v [in] Direction vector on which the Hessian is applied: \f$\boldsymbol{v}_{\boldsymbol{x}}\f$.
   * 
   * \param z [in] Lagrange multiplier vector: \f$\boldsymbol{z}\f$.
   * 
   * \param x [in] Solution vector for the current time step: \f$\boldsymbol{x}\f$.
   * 
   * \param xdot [in] Velocity vector for the current time step.
   * 
   * \param xdotdot [in] Acceleration vector for the current time step.
   * 
   * \param param_array [in] Array of the parameters vectors.
   * 
   * \param Hv_f_xx [out] the output of the computation: \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}}\f$.
   * 
   * This implementation relies on:
   * <ul>
   *  <li> PHAL::GatherSolution<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::GatherScalarNodalParameter<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::GatherScalarExtruded2DNodalParameter<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::ScatterResidual<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::ScatterResidualWithExtrudedParams<PHAL::AlbanyTraits::HessianVec,Traits>.
   * </ul>
   */
void
  evaluateResidual_HessVecProd_xx(
      const double                            current_time,
      const Teuchos::RCP<const Thyra_MultiVector>& v,
      const Teuchos::RCP<const Thyra_Vector>& z,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>&         param_array,
      const Teuchos::RCP<Thyra_MultiVector>&  Hv_f_xx);

  /**
   * \brief evaluateResidual_HessVecProd_xp function
   * 
   * This function is used to compute contributions of the application of the Hessian \f$\boldsymbol{H}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\f$ of the inner product of the residual vector \f$\boldsymbol{f}\f$
   * and a Lagrange multiplier vector \f$\boldsymbol{z}\f$ to a direction vector \f$\boldsymbol{v}\f$.
   * 
   * Representing the Hessian \f$\boldsymbol{H}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\f$ as a block matrix:
   * \f[
   *   \boldsymbol{H}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) := 
   *   \begin{bmatrix} 
   *     \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{x}^{2}} & \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{x} \partial \boldsymbol{p}_{1}} & \cdots & \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{x} \partial \boldsymbol{p}_{n}} \\
   *     \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{p}_{1} \partial \boldsymbol{x}} & \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{p}_{1}^{2}} & \cdots & \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{p}_{1} \partial \boldsymbol{p}_{n}} \\
   *     \vdots & \vdots & \ddots & \vdots\\
   *     \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{p}_{n} \partial \boldsymbol{x}} & \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{p}_{n}\partial \boldsymbol{p}_{1}} & \cdots & \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{p}_{n}^{2}}
   *   \end{bmatrix} =
   *   \begin{bmatrix} 
   *     \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) & \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) & \cdots & \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_n}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) \\ 
   *     \boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) & \boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) & \cdots & \boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{p}_n}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) \\
   *     \vdots & \vdots & \ddots & \vdots\\
   *     \boldsymbol{H}_{\boldsymbol{p}_n\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) & \boldsymbol{H}_{\boldsymbol{p}_n\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) & \cdots & \boldsymbol{H}_{\boldsymbol{p}_n\boldsymbol{p}_n}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) 
   *   \end{bmatrix},
   * \f]
   * where \f$\boldsymbol{x}\f$ is the solution vector, \f$\boldsymbol{p}_1\f$ is a first  parameter, \f$\ldots\f$, and \f$\boldsymbol{p}_n\f$ is the \f$n\f$-th 
   *  parameter, this function computes the contribution:
   * \f[
   *   \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_j}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_j},
   * \f]
   * where \f$j\f$ is an index included in the range \f$\left[1,n\right]\f$.
   * 
   * This function can currently compute only one block contribution at a time; for one function call \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_j}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_j}\f$
   * can be computed for only one fixed \f$j\f$ at a time.
   * 
   * \param current_time [in] Current time at which the Hessian is computed for transient simulations.
   * 
   * \param v [in] Direction vector on which the Hessian is applied: \f$\boldsymbol{v}_{\boldsymbol{p}_j}\f$.
   * 
   * \param z [in] Lagrange multiplier vector: \f$\boldsymbol{z}\f$.
   * 
   * \param x [in] Solution vector for the current time step: \f$\boldsymbol{x}\f$.
   * 
   * \param xdot [in] Velocity vector for the current time step.
   * 
   * \param xdotdot [in] Acceleration vector for the current time step.
   * 
   * \param param_array [in] Array of the parameters vectors.
   * 
   * \param param_direction_name [in] Name of the parameter used for the second differentiation: \f$\boldsymbol{p}_j\f$.
   * 
   * \param Hv_f_xp [out] the output of the computation: \f$\boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_j}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_j}\f$.
   * 
   * This implementation relies on:
   * <ul>
   *  <li> PHAL::GatherSolution<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::GatherScalarNodalParameter<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::GatherScalarExtruded2DNodalParameter<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::ScatterResidual<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::ScatterResidualWithExtrudedParams<PHAL::AlbanyTraits::HessianVec,Traits>.
   * </ul>
   */
  void
  evaluateResidual_HessVecProd_xp(
      const double                            current_time,
      const Teuchos::RCP<const Thyra_MultiVector>& v,
      const Teuchos::RCP<const Thyra_Vector>& z,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>&         param_array,
      const std::string&                      param_direction_name,
      const Teuchos::RCP<Thyra_MultiVector>&  Hv_f_xp);

  /**
   * \brief evaluateResidual_HessVecProd_px function
   * 
   * This function is used to compute contributions of the application of the Hessian \f$\boldsymbol{H}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\f$ of the inner product of the residual vector \f$\boldsymbol{f}\f$
   * and a Lagrange multiplier vector \f$\boldsymbol{z}\f$ to a direction vector \f$\boldsymbol{v}\f$.
   * 
   * Representing the Hessian \f$\boldsymbol{H}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\f$ as a block matrix:
   * \f[
   *   \boldsymbol{H}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) := 
   *   \begin{bmatrix} 
   *     \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{x}^{2}} & \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{x} \partial \boldsymbol{p}_{1}} & \cdots & \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{x} \partial \boldsymbol{p}_{n}} \\
   *     \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{p}_{1} \partial \boldsymbol{x}} & \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{p}_{1}^{2}} & \cdots & \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{p}_{1} \partial \boldsymbol{p}_{n}} \\
   *     \vdots & \vdots & \ddots & \vdots\\
   *     \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{p}_{n} \partial \boldsymbol{x}} & \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{p}_{n}\partial \boldsymbol{p}_{1}} & \cdots & \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{p}_{n}^{2}}
   *   \end{bmatrix} =
   *   \begin{bmatrix} 
   *     \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) & \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) & \cdots & \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_n}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) \\ 
   *     \boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) & \boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) & \cdots & \boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{p}_n}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) \\
   *     \vdots & \vdots & \ddots & \vdots\\
   *     \boldsymbol{H}_{\boldsymbol{p}_n\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) & \boldsymbol{H}_{\boldsymbol{p}_n\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) & \cdots & \boldsymbol{H}_{\boldsymbol{p}_n\boldsymbol{p}_n}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) 
   *   \end{bmatrix},
   * \f]
   * where \f$\boldsymbol{x}\f$ is the solution vector, \f$\boldsymbol{p}_1\f$ is a first  parameter, \f$\ldots\f$, and \f$\boldsymbol{p}_n\f$ is the \f$n\f$-th 
   *  parameter, this function computes the contribution:
   * \f[
   *   \boldsymbol{H}_{\boldsymbol{p}_i\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}},
   * \f]
   * where \f$i\f$ is an index included in the range \f$\left[1,n\right]\f$.
   * 
   * This function can currently compute only one block contribution at a time; for one function call \f$\boldsymbol{H}_{\boldsymbol{p}_i\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}}\f$
   * can be computed for only one fixed \f$i\f$ at a time.
   * 
   * \param current_time [in] Current time at which the Hessian is computed for transient simulations.
   * 
   * \param v [in] Direction vector on which the Hessian is applied: \f$\boldsymbol{v}_{\boldsymbol{x}}\f$.
   * 
   * \param z [in] Lagrange multiplier vector: \f$\boldsymbol{z}\f$.
   * 
   * \param x [in] Solution vector for the current time step: \f$\boldsymbol{x}\f$.
   * 
   * \param xdot [in] Velocity vector for the current time step.
   * 
   * \param xdotdot [in] Acceleration vector for the current time step.
   * 
   * \param param_array [in] Array of the parameters vectors.
   * 
   * \param param_name [in] Name of the parameter used for the first differentiation: \f$\boldsymbol{p}_i\f$.
   * 
   * \param Hv_f_px [out] the output of the computation: \f$\boldsymbol{H}_{\boldsymbol{p}_i\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{x}}\f$.
   * 
   * This implementation relies on:
   * <ul>
   *  <li> PHAL::GatherSolution<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::GatherScalarNodalParameter<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::GatherScalarExtruded2DNodalParameter<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::ScatterResidual<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::ScatterResidualWithExtrudedParams<PHAL::AlbanyTraits::HessianVec,Traits>.
   * </ul>
   */
  void
  evaluateResidual_HessVecProd_px(
      const double                            current_time,
      const Teuchos::RCP<const Thyra_MultiVector>& v,
      const Teuchos::RCP<const Thyra_Vector>& z,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>&         param_array,
      const std::string&                      param_name,
      const Teuchos::RCP<Thyra_MultiVector>&  Hv_f_px);

  /**
   * \brief evaluateResidual_HessVecProd_pp function
   * 
   * This function is used to compute contributions of the application of the Hessian \f$\boldsymbol{H}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\f$ of the inner product of the residual vector \f$\boldsymbol{f}\f$
   * and a Lagrange multiplier vector \f$\boldsymbol{z}\f$ to a direction vector \f$\boldsymbol{v}\f$.
   * 
   * Representing the Hessian \f$\boldsymbol{H}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\f$ as a block matrix:
   * \f[
   *   \boldsymbol{H}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) := 
   *   \begin{bmatrix} 
   *     \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{x}^{2}} & \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{x} \partial \boldsymbol{p}_{1}} & \cdots & \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{x} \partial \boldsymbol{p}_{n}} \\
   *     \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{p}_{1} \partial \boldsymbol{x}} & \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{p}_{1}^{2}} & \cdots & \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{p}_{1} \partial \boldsymbol{p}_{n}} \\
   *     \vdots & \vdots & \ddots & \vdots\\
   *     \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{p}_{n} \partial \boldsymbol{x}} & \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{p}_{n}\partial \boldsymbol{p}_{1}} & \cdots & \frac{\partial^{2} \left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle}{\partial \boldsymbol{p}_{n}^{2}}
   *   \end{bmatrix} =
   *   \begin{bmatrix} 
   *     \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) & \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) & \cdots & \boldsymbol{H}_{\boldsymbol{x}\boldsymbol{p}_n}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) \\ 
   *     \boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) & \boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) & \cdots & \boldsymbol{H}_{\boldsymbol{p}_1\boldsymbol{p}_n}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) \\
   *     \vdots & \vdots & \ddots & \vdots\\
   *     \boldsymbol{H}_{\boldsymbol{p}_n\boldsymbol{x}}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) & \boldsymbol{H}_{\boldsymbol{p}_n\boldsymbol{p}_1}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) & \cdots & \boldsymbol{H}_{\boldsymbol{p}_n\boldsymbol{p}_n}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle) 
   *   \end{bmatrix},
   * \f]
   * where \f$\boldsymbol{x}\f$ is the solution vector, \f$\boldsymbol{p}_1\f$ is a first  parameter, \f$\ldots\f$, and \f$\boldsymbol{p}_n\f$ is the \f$n\f$-th 
   *  parameter, this function computes the contribution:
   * \f[
   *   \boldsymbol{H}_{\boldsymbol{p}_i\boldsymbol{p}_j}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_j},
   * \f]
   * where \f$i\f$ and \f$j\f$ are 2 (potentially different) indices included in the range \f$\left[1,n\right]\f$.
   * 
   * This function can currently compute only one block contribution at a time; for one function call \f$\boldsymbol{H}_{\boldsymbol{p}_i\boldsymbol{p}_j}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_j}\f$
   * can be computed for only one fixed \f$i\f$ and one fixed \f$j\f$ at a time.
   * 
   * \param current_time [in] Current time at which the Hessian is computed for transient simulations.
   * 
   * \param v [in] Direction vector on which the Hessian is applied: \f$\boldsymbol{v}_{\boldsymbol{p}_j}\f$.
   * 
   * \param z [in] Lagrange multiplier vector: \f$\boldsymbol{z}\f$.
   * 
   * \param x [in] Solution vector for the current time step: \f$\boldsymbol{x}\f$.
   * 
   * \param xdot [in] Velocity vector for the current time step.
   * 
   * \param xdotdot [in] Acceleration vector for the current time step.
   * 
   * \param param_array [in] Array of the parameters vectors.
   * 
   * \param param_name [in] Name of the parameter used for the first differentiation: \f$\boldsymbol{p}_i\f$.
   * 
   * \param param_direction_name [in] Name of the parameter used for the second differentiation: \f$\boldsymbol{p}_j\f$.
   * 
   * \param Hv_f_pp [out] the output of the computation: \f$\boldsymbol{H}_{\boldsymbol{p}_i\boldsymbol{p}_j}(\left\langle \boldsymbol{f},\boldsymbol{z}\right\rangle)\boldsymbol{v}_{\boldsymbol{p}_j}\f$.
   * 
   * This implementation relies on:
   * <ul>
   *  <li> PHAL::GatherSolution<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::GatherScalarNodalParameter<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::GatherScalarExtruded2DNodalParameter<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::ScatterResidual<PHAL::AlbanyTraits::HessianVec,Traits>,
   *  <li> PHAL::ScatterResidualWithExtrudedParams<PHAL::AlbanyTraits::HessianVec,Traits>.
   * </ul>
   */
  void
  evaluateResidual_HessVecProd_pp(
      const double                            current_time,
      const Teuchos::RCP<const Thyra_MultiVector>& v,
      const Teuchos::RCP<const Thyra_Vector>& z,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>&         param_array,
      const std::string&                      param_name,
      const std::string&                      param_direction_name,
      const Teuchos::RCP<Thyra_MultiVector>&  Hv_f_pp);

  //! Provide access to shapeParameters -- no AD
  PHAL::AlbanyTraits::Residual::ScalarT&
  getValue(const std::string& n);

  //! Class to manage state variables (a.k.a. history)
  StateManager&
  getStateMgr()
  {
    return stateMgr;
  }

  //! Evaluate state field manager
  void
  evaluateStateFieldManager(
      const double                     current_time,
      const Thyra_Vector&              x,
      Teuchos::Ptr<const Thyra_Vector> xdot,
      Teuchos::Ptr<const Thyra_Vector> xdotdot,
      Teuchos::Ptr<const Thyra_MultiVector> dxdp = Teuchos::null);

  void
  evaluateStateFieldManager(
      const double             current_time,
      const Thyra_MultiVector& x,
      Teuchos::Ptr<const Thyra_MultiVector> dxdp = Teuchos::null);

  //! Access to number of worksets - needed for working with StateManager
  int
  getNumWorksets()
  {
    return disc->getWsElNodeEqID().size();
  }

  //! Const access to problem parameter list
  Teuchos::RCP<const Teuchos::ParameterList>
  getProblemPL() const
  {
    return problemParams;
  }

  //! Access to problem parameter list
  Teuchos::RCP<Teuchos::ParameterList>
  getProblemPL()
  {
    return problemParams;
  }

  //! Const access to app parameter list
  Teuchos::RCP<const Teuchos::ParameterList>
  getAppPL() const
  {
    return params_;
  }

  //! Access to app parameter list
  Teuchos::RCP<Teuchos::ParameterList>
  getAppPL()
  {
    return params_;
  }

  bool is_adjoint;

 private:
  //! Utility function to set up ShapeParameters through Sacado
  void
  registerShapeParameters();

  void
  defineTimers();

  void
  removeEpetraRelatedPLs(const Teuchos::RCP<Teuchos::ParameterList>& params);

 public:
  //! Routine to get workset (bucket) size info needed by all Evaluation types
  template <typename EvalT>
  void
  loadWorksetBucketInfo(PHAL::Workset& workset, const int& ws,
      const std::string& evalName);

  void
  loadBasicWorksetInfo(PHAL::Workset& workset, double current_time);

  void
  loadBasicWorksetInfoSDBCs(
      PHAL::Workset&                          workset,
      const Teuchos::RCP<const Thyra_Vector>& owned_sol,
      const double                            current_time);

  void
  loadWorksetJacobianInfo(
      PHAL::Workset& workset,
      const double   alpha,
      const double   beta,
      const double   omega);

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>>
  getEnrichedMeshSpecs() const
  {
    return meshSpecs;
  }

  //! Routine to load common nodeset info into workset
  void
  loadWorksetNodesetInfo(PHAL::Workset& workset);

  //! Routine to load common sideset info into workset
  void
  loadWorksetSidesetInfo(PHAL::Workset& workset, const int ws);

  //! Routines for setting a scaling to be applied to the Jacobian/resdiual
  void
  setScale(Teuchos::RCP<const Thyra_LinearOp> jac = Teuchos::null);
  void
  setScaleBCDofs(
      PHAL::Workset&                     workset,
      Teuchos::RCP<const Thyra_LinearOp> jac = Teuchos::null);

  void
  setupBasicWorksetInfo(
      PHAL::Workset&                          workset,
      double                                  current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>&         p);

  void
  setupTangentWorksetInfo(
      PHAL::Workset&                               workset,
      double                                       current_time,
      bool                                         sum_derivs,
      const Teuchos::RCP<const Thyra_Vector>&      x,
      const Teuchos::RCP<const Thyra_Vector>&      xdot,
      const Teuchos::RCP<const Thyra_Vector>&      xdotdot,
      const Teuchos::Array<ParamVec>&              p,
      ParamVec*                                    deriv_p,
      const Teuchos::RCP<const Thyra_MultiVector>& Vx,
      const Teuchos::RCP<const Thyra_MultiVector>& Vxdot,
      const Teuchos::RCP<const Thyra_MultiVector>& Vxdotdot,
      const Teuchos::RCP<const Thyra_MultiVector>& Vp);

  int
  calcTangentDerivDimension(const Teuchos::RCP<Teuchos::ParameterList>& problemParams);

 private:
  template <typename EvalT>
  void
  postRegSetup();

  template <typename EvalT>
  void
  postRegSetupDImpl();

  template <typename EvalT>
  void
  writePhalanxGraph(Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>> fm,
      const std::string& evalName, const int& phxGraphVisDetail);

 public:
  double
  fixTime(double const current_time) const
  {
    bool const use_time_param = paramLib->isParameter("Time") == true;

    double const this_time =
        use_time_param == true ?
            paramLib->getRealValue<PHAL::AlbanyTraits::Residual>("Time") :
            current_time;

    return this_time;
  }

  void
  setScaling(const Teuchos::RCP<Teuchos::ParameterList>& params);

 public:
  //! Get Phalanx postRegistration data
  Teuchos::RCP<PHAL::Setup>
  getPhxSetup()
  {
    return phxSetup;
  }

 protected:
  bool no_dir_bcs_;
  bool requires_sdbcs_;
  bool requires_orig_dbcs_;

  //! Teuchos communicator
  Teuchos::RCP<const Teuchos_Comm> comm;

  //! Output stream, defaults to pronting just Proc 0
  Teuchos::RCP<Teuchos::FancyOStream> out;

  //! Element discretization
  Teuchos::RCP<Albany::AbstractDiscretization> disc;

  //! discretization factory
  Teuchos::RCP<Albany::DiscretizationFactory> discFactory;

  //! mesh specs
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>> meshSpecs;

  //! Problem class
  Teuchos::RCP<Albany::AbstractProblem> problem;

  //! Problem Parameters
  Teuchos::RCP<Teuchos::ParameterList> problemParams;

  //! App Parameters
  Teuchos::RCP<Teuchos::ParameterList> params_;

  //! Parameter library
  Teuchos::RCP<ParamLib> paramLib;

  //! Distributed parameter library
  Teuchos::RCP<DistributedParameterLibrary> distParamLib;

  //! Solution memory manager
  Teuchos::RCP<Albany::SolutionManager> solMgr;

  //! Response functions
  Teuchos::Array<Teuchos::RCP<Albany::AbstractResponseFunction>> responses;

  //! Phalanx Field Manager for volumetric fills
  Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>> fm;

  //! Phalanx Field Manager for Dirichlet Conditions
  Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>> dfm;

  //! Phalanx Field Manager for Neumann Conditions
  Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>> nfm;

  //! Phalanx Field Manager for states
  Teuchos::Array<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>> sfm;

  bool explicit_scheme;

  //! Data for Physics-Based Preconditioners
  bool                                 physicsBasedPreconditioner;
  Teuchos::RCP<Teuchos::ParameterList> precParams;
  std::string                          precType;

  //! Type of solution method
  SolutionMethod solMethod;

  //! Integer specifying whether user wants to write Jacobian to MatrixMarket
  //! file
  // writeToMatrixMarketJac = 0: no writing to MatrixMarket (default)
  // writeToMatrixMarketJac =-1: write to MatrixMarket every time a Jacobian
  // arises writeToMatrixMarketJac = N: write N^th Jacobian to MatrixMarket
  // ...and similarly for writeToMatrixMarketRes (integer specifying whether
  // user wants to write residual to MatrixMarket file) and writeToMatrixMarketSoln
  // (integer specifying whether user wants to write solution to MatrixMarket file)
  int writeToMatrixMarketJac;
  int writeToMatrixMarketRes;
  int writeToMatrixMarketSoln;
  int computeJacCondNum;
  //! Integer specifying whether user wants to write Jacobian, residual and solution to
  //! Standard output (cout)
  int writeToCoutJac;
  int writeToCoutRes;
  int writeToCoutSoln;

  // Value to scale Jacobian/Residual by to possibly improve conditioning
  double scale;
  double scaleBCdofs;
  // Scaling types
  enum SCALETYPE
  {
    CONSTANT,
    DIAG,
    ABSROWSUM
  };
  SCALETYPE scale_type;

  //! Shape Optimization data
  bool                     shapeParamsHaveBeenReset;
  std::vector<RealType>    shapeParams;
  std::vector<std::string> shapeParamNames;

  unsigned int neq, spatial_dimension, tangent_deriv_dim;

  //! Phalanx postRegistration data
  Teuchos::RCP<PHAL::Setup> phxSetup;
  mutable int               phxGraphVisDetail;
  mutable int               stateGraphVisDetail;

  StateManager stateMgr;

  bool morphFromInit;
  bool ignore_residual_in_jacobian;

  //! To prevent a singular mass matrix associated with Dirichlet
  //  conditions, optionally add a small perturbation to the diag
  double perturbBetaForDirichlets;

  void
  determinePiroSolver(
      const Teuchos::RCP<Teuchos::ParameterList>& topLevelParams);

  int derivatives_check_;

  int num_time_deriv;

  // The following are for Jacobian/residual scaling
  Teuchos::Array<Teuchos::Array<int>> offsets_;
  std::vector<std::string>            nodeSetIDs_;
  Teuchos::RCP<Thyra_Vector>          scaleVec_;

  // boolean read from input file telling code whether to compute/print
  // responses every step
  bool observe_responses;

  // how often one wants the responses to be computed/printed
  int response_observ_freq;

  // local responses
  Teuchos::Array<unsigned int> relative_responses;
};

template <typename EvalT>
void
Application::loadWorksetBucketInfo(PHAL::Workset& workset, const int& ws,
    const std::string& evalName)
{
  auto const& wsElNodeEqID       = disc->getWsElNodeEqID();
  auto const& wsElNodeID         = disc->getWsElNodeID();
  auto const& coords             = disc->getCoords();
  auto const& wsEBNames          = disc->getWsEBNames();

  workset.numCells             = wsElNodeEqID[ws].extent(0);
  workset.wsElNodeEqID         = wsElNodeEqID[ws];
  workset.wsElNodeID           = wsElNodeID[ws];
  workset.wsCoords             = coords[ws];
  workset.EBName               = wsEBNames[ws];
  workset.wsIndex              = ws;

  workset.local_Vp.resize(workset.numCells);

  workset.savedMDFields = phxSetup->get_saved_fields(evalName);

  //  workset.print(*out);

  // Sidesets are integrated within the Cells
  loadWorksetSidesetInfo(workset, ws);

  workset.stateArrayPtr =
      &stateMgr.getStateArray(Albany::StateManager::ELEM, ws);
#if defined(ALBANY_EPETRA)
  workset.disc         = disc;  // Needed by LandIce for sideset DOF save
  workset.eigenDataPtr = stateMgr.getEigenData();
  workset.auxDataPtr   = stateMgr.getAuxData();
#endif
  // FIXME, 6/25: This line was causing link error.  Need to figure out why.
  // workset.auxDataPtrT = stateMgr.getAuxDataT();

  auto blocked_disc = Teuchos::rcp_dynamic_cast<BlockedSTKDiscretization>(disc);
  if (!blocked_disc.is_null()) {
    workset.blockedWsElNodeEqID.resize(blocked_disc->numBlocks());
    for (int i=0; i<blocked_disc->numBlocks(); ++i) {
      workset.blockedWsElNodeEqID[i] = blocked_disc->getBlock(i)->getWsElNodeEqID()[ws];
    }
  }
}

}  // namespace Albany

#endif  // ALBANY_APPLICATION_HPP
