//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_WORKSET_HPP
#define PHAL_WORKSET_HPP

#include <list>
#include <set>
#include <string>

#include "Albany_SacadoTypes.hpp"
#include "Albany_ThyraTypes.hpp"
#include "Albany_TpetraTypes.hpp"
#include "PHAL_Setup.hpp"

#include "Albany_DiscretizationUtils.hpp"
#include "Albany_StateInfoStruct.hpp"

#include "Kokkos_ViewFactory.hpp"

#include "Teuchos_Comm.hpp"
#include "Teuchos_RCP.hpp"

// Forward declarations
namespace Albany {
class AbstractDiscretization;
class CombineAndScatterManager;
class DistributedParameterLibrary;

#if defined(ALBANY_EPETRA)
struct EigendataStruct;
#endif
}  // namespace Albany
#if defined(ALBANY_EPETRA)
class Epetra_MultiVector;
#endif

namespace PHAL {

struct HessianWorkset
{
  Teuchos::RCP<Thyra_MultiVector> direction_x;
  Teuchos::RCP<Thyra_MultiVector> direction_p;

  Teuchos::RCP<Thyra_Vector> f_multiplier;
  Teuchos::RCP<Thyra_Vector> overlapped_f_multiplier;

  Teuchos::RCP<Thyra_MultiVector> hess_vec_prod_f_xx;
  Teuchos::RCP<Thyra_MultiVector> hess_vec_prod_f_xp;
  Teuchos::RCP<Thyra_MultiVector> hess_vec_prod_f_px;
  Teuchos::RCP<Thyra_MultiVector> hess_vec_prod_f_pp;

  Teuchos::RCP<Thyra_MultiVector> hess_vec_prod_g_xx;
  Teuchos::RCP<Thyra_MultiVector> hess_vec_prod_g_xp;
  Teuchos::RCP<Thyra_MultiVector> hess_vec_prod_g_px;
  Teuchos::RCP<Thyra_MultiVector> hess_vec_prod_g_pp;

  Teuchos::RCP<Thyra_MultiVector> overlapped_hess_vec_prod_f_xx;
  Teuchos::RCP<Thyra_MultiVector> overlapped_hess_vec_prod_f_xp;
  Teuchos::RCP<Thyra_MultiVector> overlapped_hess_vec_prod_f_px;
  Teuchos::RCP<Thyra_MultiVector> overlapped_hess_vec_prod_f_pp;

  Teuchos::RCP<Thyra_MultiVector> overlapped_hess_vec_prod_g_xx;
  Teuchos::RCP<Thyra_MultiVector> overlapped_hess_vec_prod_g_xp;
  Teuchos::RCP<Thyra_MultiVector> overlapped_hess_vec_prod_g_px;
  Teuchos::RCP<Thyra_MultiVector> overlapped_hess_vec_prod_g_pp;

  std::string dist_param_deriv_direction_name;
  Teuchos::RCP<const Albany::CombineAndScatterManager> p_direction_cas_manager;
};

struct Workset
{
  Workset()
      : transientTerms(false), accelerationTerms(false), ignore_residual(false)
  {
  }

  unsigned int numCells;
  unsigned int wsIndex;
  unsigned int numEqs;

  // Solution vector (and time derivatives)
  Teuchos::RCP<const Thyra_Vector> x;
  Teuchos::RCP<const Thyra_Vector> xdot;
  Teuchos::RCP<const Thyra_Vector> xdotdot;

  Teuchos::RCP<ParamVec> params;

  // Component of Tangent vector direction along x, xdot, xdotdot, and p.
  // These are used to compute df/dx*Vx + df/dxdot*Vxdot + df/dxdotdot*Vxdotdot
  // + df/dp*Vp.
  Teuchos::RCP<const Thyra_MultiVector> Vx;
  Teuchos::RCP<const Thyra_MultiVector> Vxdot;
  Teuchos::RCP<const Thyra_MultiVector> Vxdotdot;
  Teuchos::RCP<const Thyra_MultiVector> Vp;

  // These are residual related.
  Teuchos::RCP<Thyra_Vector>      f;
  Teuchos::RCP<Thyra_LinearOp>    Jac;
  Teuchos::RCP<Thyra_MultiVector> JV;
  Teuchos::RCP<Thyra_MultiVector> fp;
  Teuchos::RCP<Thyra_MultiVector> fpV;
  Teuchos::RCP<Thyra_MultiVector> Vp_bc;

  Albany::DeviceView1d<ST>        f_kokkos;
  Albany::DeviceLocalMatrix<ST>   Jac_kokkos;

  Teuchos::RCP<const Albany::NodeSetList>      nodeSets;
  Teuchos::RCP<const Albany::NodeSetCoordList> nodeSetCoords;

  Teuchos::RCP<const Albany::SideSetList> sideSets;
  Teuchos::RCP<const Albany::LocalSideSetInfoList> sideSetViews;

  Teuchos::RCP<const std::map<std::string, Kokkos::View<LO****, PHX::Device>>> localDOFViews;

  // jacobian and mass matrix coefficients for matrix fill
  double j_coeff;
  double m_coeff;  // d(x_dot)/dx_{new}
  double n_coeff;  // d(x_dotdot)/dx_{new}

  // Current Time as defined by Tempus
  double current_time;
  // amb Nowhere set. We should either set it or remove it.
  double previous_time;

  double time_step;

  // flag indicating whether to sum tangent derivatives, i.e.,
  // compute alpha*df/dxdot*Vxdot + beta*df/dx*Vx + omega*df/dxddotot*Vxdotdot +
  // df/dp*Vp or compute alpha*df/dxdot*Vxdot + beta*df/dx*Vx +
  // omega*df/dxdotdot*Vxdotdot and df/dp*Vp separately
  int num_cols_x;
  int num_cols_p;
  int param_offset;

  // Distributed parameter derivatives
  Teuchos::RCP<Albany::DistributedParameterLibrary> distParamLib;
  std::string                                       dist_param_deriv_name;
  bool                                              transpose_dist_param_deriv;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double>>> local_Vp;

  std::vector<PHX::index_size_type> Jacobian_deriv_dims;
  std::vector<PHX::index_size_type> Tangent_deriv_dims;

  Teuchos::Array<Albany::WorksetConn>           blockedWsElNodeEqID;

  Albany::WorksetConn                           wsElNodeEqID;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO>>      wsElNodeID;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*>> wsCoords;
  std::string                                   EBName;

  // Needed for Schwarz coupling and for dirichlet conditions based on dist
  // parameters.
  Teuchos::RCP<Albany::AbstractDiscretization> disc;

  int spatial_dimension_{0};

  Albany::StateArray* stateArrayPtr;
#if defined(ALBANY_EPETRA)
  Teuchos::RCP<Albany::EigendataStruct> eigenDataPtr;
  Teuchos::RCP<Epetra_MultiVector>      auxDataPtr;
#endif
  // Teuchos::RCP<Albany::EigendataStructT> eigenDataPtrT;
  Teuchos::RCP<Tpetra_MultiVector> auxDataPtrT;

  bool transientTerms;
  bool accelerationTerms;

  // Flag indicating whether to ignore residual calculations in the
  // Jacobian calculation.  This only works for some problems where the
  // the calculation of the Jacobian doesn't require calculation of the
  // residual (such as linear problems), but if it does work it can
  // significantly reduce Jacobian calculation cost.
  bool ignore_residual;

  // Flag indicated whether we are solving the adjoint operator or the
  // forward operator.  This is used in the Albany application when
  // either the Jacobian or the transpose of the Jacobian is scattered.
  bool is_adjoint;

  // New field manager response stuff
  Teuchos::RCP<const Teuchos::Comm<int>> comm;

  // Combine and Scatter manager (for import-export of responses derivatives),
  // for both solution (x) and distributed parameter (p)
  Teuchos::RCP<const Albany::CombineAndScatterManager> x_cas_manager;
  Teuchos::RCP<const Albany::CombineAndScatterManager> p_cas_manager;

  // Response vector g and its derivatives
  Teuchos::RCP<Thyra_Vector>      g;
  Teuchos::RCP<Thyra_MultiVector> dgdx;
  Teuchos::RCP<Thyra_MultiVector> dgdxdot;
  Teuchos::RCP<Thyra_MultiVector> dgdxdotdot;
  Teuchos::RCP<Thyra_MultiVector> dgdp;

  // Overlapped version of response derivatives
  Teuchos::RCP<Thyra_MultiVector> overlapped_dgdx;
  Teuchos::RCP<Thyra_MultiVector> overlapped_dgdxdot;
  Teuchos::RCP<Thyra_MultiVector> overlapped_dgdxdotdot;
  Teuchos::RCP<Thyra_MultiVector> overlapped_dgdp;

  // List of saved MDFields (needed for memoization)
  Teuchos::RCP<const StringSet> savedMDFields;

  HessianWorkset hessianWorkset;

  // Meta-function class encoding T<EvalT::ScalarT> given EvalT
  // where T is any lambda expression (typically a placeholder expression)
  template <typename T>
  struct ApplyEvalT
  {
    template <typename EvalT>
    struct apply
    {
      typedef
          typename Sacado::mpl::apply<T, typename EvalT::ScalarT>::type type;
    };
  };

  // Meta-function class encoding RCP<ValueTypeSerializer<int,T> > for a given
  // type T.  This is to eliminate an error when using a placeholder expression
  // for the same thing in CreateLambdaKeyMap below
  struct ApplyVTS
  {
    template <typename T>
    struct apply
    {
      typedef Teuchos::RCP<Teuchos::ValueTypeSerializer<int, T>> type;
    };
  };

  void
  print(std::ostream& os)
  {
    os << "Printing workset data:" << std::endl;
    os << "\tEB name : " << EBName << std::endl;
    os << "\tnumCells : " << numCells << std::endl;
    os << "\twsElNodeEqID : " << std::endl;
    for (unsigned int i = 0; i < wsElNodeEqID.extent(0); i++)
      for (unsigned int j = 0; j < wsElNodeEqID.extent(1); j++)
        for (unsigned int k = 0; k < wsElNodeEqID.extent(2); k++)
          os << "\t\twsElNodeEqID(" << i << "," << j << "," << k
             << ") = " << wsElNodeEqID(i, j, k) << std::endl;
    os << "\twsCoords : " << std::endl;
    for (int i = 0; i < wsCoords.size(); i++)
      for (int j = 0; j < wsCoords[i].size(); j++)
        os << "\t\tcoord0:" << wsCoords[i][j][0] << "][" << wsCoords[i][j][1]
           << std::endl;
  }
};

}  // namespace PHAL

#endif  // PHAL_WORKSET_HPP
