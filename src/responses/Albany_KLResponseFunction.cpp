//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/13/14: Epetra ifdef'ed out except SG and MP when ALBANY_EPETRA_EXE is off.

#include "Albany_KLResponseFunction.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_VerboseObject.hpp"

Albany::KLResponseFunction::
KLResponseFunction(
  const Teuchos::RCP<Albany::AbstractResponseFunction>& response_,
  Teuchos::ParameterList& responseParams) :
  response(response_),
  responseParams(responseParams),
  out(Teuchos::VerboseObjectBase::getDefaultOStream())
{
  num_kl = responseParams.get("Number of KL Terms", 5);
}

Albany::KLResponseFunction::
~KLResponseFunction()
{
}

#if defined(ALBANY_EPETRA)
Teuchos::RCP<const Epetra_Map>
Albany::KLResponseFunction::
responseMap() const
{
  return response->responseMap();
}
#endif

Teuchos::RCP<const Tpetra_Map>
Albany::KLResponseFunction::
responseMapT() const
{
  return response->responseMapT();
}

#if defined(ALBANY_EPETRA)
Teuchos::RCP<Epetra_Operator>
Albany::KLResponseFunction::
createGradientOp() const
{
  return response->createGradientOp();
}
#endif

Teuchos::RCP<Tpetra_Operator>
Albany::KLResponseFunction::
createGradientOpT() const
{
  return response->createGradientOpT();
}

bool
Albany::KLResponseFunction::
isScalarResponse() const
{
  return response->isScalarResponse();
}


void
Albany::KLResponseFunction::
evaluateResponseT(const double current_time,
     const Tpetra_Vector* xdotT,
     const Tpetra_Vector* xdotdotT,
     const Tpetra_Vector& xT,
     const Teuchos::Array<ParamVec>& p,
     Tpetra_Vector& gT)
{
  response->evaluateResponseT(current_time, xdotT, xdotdotT, xT, p, gT);
}


void
Albany::KLResponseFunction::
evaluateTangentT(const double alpha,
    const double beta,
    const double omega,
    const double current_time,
    bool sum_derivs,
    const Tpetra_Vector* xdotT,
    const Tpetra_Vector* xdotdotT,
    const Tpetra_Vector& xT,
    const Teuchos::Array<ParamVec>& p,
    ParamVec* deriv_p,
    const Tpetra_MultiVector* VxdotT,
    const Tpetra_MultiVector* VxdotdotT,
    const Tpetra_MultiVector* VxT,
    const Tpetra_MultiVector* VpT,
    Tpetra_Vector* gT,
    Tpetra_MultiVector* gxT,
    Tpetra_MultiVector* gpT)
{
  response->evaluateTangentT(alpha, beta, omega, current_time, sum_derivs,
          xdotT, xdotdotT, xT, p, deriv_p, VxdotT, VxdotdotT, VxT, VpT,
          gT, gxT, gpT);
}

//! Evaluate distributed parameter derivative dg/dp
void
Albany::KLResponseFunction::
evaluateDistParamDerivT(
  const double current_time,
  const Tpetra_Vector* xdotT,
  const Tpetra_Vector* xdotdotT,
  const Tpetra_Vector& xT,
  const Teuchos::Array<ParamVec>& param_array,
  const std::string& dist_param_name,
  Tpetra_MultiVector*  dg_dpT){
  response->evaluateDistParamDerivT(current_time, xdotT, xdotdotT, xT, param_array, dist_param_name, dg_dpT);
}

#if defined(ALBANY_EPETRA)
void
Albany::KLResponseFunction::
evaluateDerivative(const double current_time,
       const Epetra_Vector* xdot,
       const Epetra_Vector* xdotdot,
       const Epetra_Vector& x,
       const Teuchos::Array<ParamVec>& p,
       ParamVec* deriv_p,
       Epetra_Vector* g,
       const EpetraExt::ModelEvaluator::Derivative& dg_dx,
       const EpetraExt::ModelEvaluator::Derivative& dg_dxdot,
       const EpetraExt::ModelEvaluator::Derivative& dg_dxdotdot,
       const EpetraExt::ModelEvaluator::Derivative& dg_dp)
{
  response->evaluateDerivative(current_time, xdot, xdotdot, x, p, deriv_p,
             g, dg_dx, dg_dxdot, dg_dxdotdot, dg_dp);
}
#endif

void
Albany::KLResponseFunction::
evaluateDerivativeT(const double current_time,
       const Tpetra_Vector* xdotT,
       const Tpetra_Vector* xdotdotT,
       const Tpetra_Vector& xT,
       const Teuchos::Array<ParamVec>& p,
       ParamVec* deriv_p,
       Tpetra_Vector* gT,
       const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxT,
       const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdotT,
       const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdotdotT,
       const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dpT)
{
  response->evaluateDerivativeT(current_time, xdotT, xdotdotT, xT, p, deriv_p,
             gT, dg_dxT, dg_dxdotT, dg_dxdotdotT, dg_dpT);
}

#ifdef ALBANY_ENSEMBLE

void
Albany::KLResponseFunction::
evaluateMPResponse(const double curr_time,
       const Stokhos::ProductEpetraVector* mp_xdot,
       const Stokhos::ProductEpetraVector* mp_xdotdot,
       const Stokhos::ProductEpetraVector& mp_x,
       const Teuchos::Array<ParamVec>& p,
       const Teuchos::Array<int>& mp_p_index,
       const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
       Stokhos::ProductEpetraVector& mp_g)
{
  response->evaluateMPResponse(curr_time, mp_xdot, mp_xdotdot, mp_x, p, mp_p_index,
             mp_p_vals, mp_g);
}


void
Albany::KLResponseFunction::
evaluateMPTangent(const double alpha,
      const double beta,
      const double omega,
      const double current_time,
      bool sum_derivs,
      const Stokhos::ProductEpetraVector* mp_xdot,
      const Stokhos::ProductEpetraVector* mp_xdotdot,
      const Stokhos::ProductEpetraVector& mp_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& mp_p_index,
      const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
      ParamVec* deriv_p,
      const Epetra_MultiVector* Vx,
      const Epetra_MultiVector* Vxdot,
      const Epetra_MultiVector* Vxdotdot,
      const Epetra_MultiVector* Vp,
      Stokhos::ProductEpetraVector* mp_g,
      Stokhos::ProductEpetraMultiVector* mp_JV,
      Stokhos::ProductEpetraMultiVector* mp_gp)
{
  response->evaluateMPTangent(alpha, beta, omega, current_time, sum_derivs,
            mp_xdot, mp_xdotdot, mp_x, p, mp_p_index, mp_p_vals, deriv_p,
            Vxdot, Vxdotdot, Vx, Vp,
            mp_g, mp_JV, mp_gp);
}

void
Albany::KLResponseFunction::
evaluateMPDerivative(
  const double current_time,
  const Stokhos::ProductEpetraVector* mp_xdot,
  const Stokhos::ProductEpetraVector* mp_xdotdot,
  const Stokhos::ProductEpetraVector& mp_x,
  const Teuchos::Array<ParamVec>& p,
  const Teuchos::Array<int>& mp_p_index,
  const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
  ParamVec* deriv_p,
  Stokhos::ProductEpetraVector* mp_g,
  const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dx,
  const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dxdot,
  const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dxdotdot,
  const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dp)
{
  response->evaluateMPDerivative(current_time, mp_xdot, mp_xdotdot, mp_x, p, mp_p_index,
         mp_p_vals, deriv_p,
         mp_g, mp_dg_dx, mp_dg_dxdot, mp_dg_dxdotdot, mp_dg_dp);
}
#endif
