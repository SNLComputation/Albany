//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_STOKES_FO_BASE_HPP
#define LANDICE_STOKES_FO_BASE_HPP

#include "LandIce_GatherVerticallyContractedSolution.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Phalanx_Print.hpp"

#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"

#include "PHAL_AddNoise.hpp"
#include "PHAL_FieldFrobeniusNorm.hpp"
#include "PHAL_LoadSideSetStateField.hpp"
#include "PHAL_LoadStateField.hpp"
#include "PHAL_SaveSideSetStateField.hpp"
#include "PHAL_SaveStateField.hpp"

#include "Albany_AbstractProblem.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_GeneralPurposeFieldsNames.hpp"
#include "LandIce_ResponseUtilities.hpp"

#include "LandIce_BasalFrictionCoefficient.hpp"
#include "LandIce_DOFDivInterpolationSide.hpp"
#include "LandIce_EffectivePressure.hpp"
#include "LandIce_FlowRate.hpp"
#include "LandIce_FluxDiv.hpp"
#include "LandIce_IceOverburden.hpp"
#include "LandIce_ParamEnum.hpp"
#include "LandIce_ProblemUtils.hpp"
#include "PHAL_SharedParameter.hpp"
#include "LandIce_StokesFOBasalResid.hpp"
#include "LandIce_StokesFOLateralResid.hpp"
#include "LandIce_StokesFOResid.hpp"
#ifdef CISM_HAS_LANDICE
#include "LandIce_CismSurfaceGradFO.hpp"
#endif
#include "LandIce_StokesFOBodyForce.hpp"
#include "LandIce_StokesFOStress.hpp"
#include "LandIce_Time.hpp"
#include "LandIce_ViscosityFO.hpp"
#include "LandIce_Dissipation.hpp"
#include "LandIce_UpdateZCoordinate.hpp"

#include <string.hpp> // For util::upper_case (do not confuse this with <string>! string.hpp is an Albany file)

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace LandIce
{

/*!
 *  Base class for all StokesFO* problems.
 *
 *  This class implements some methods that are used across all StokesFO* problems,
 *  so to reduce code duplication. In particular, this class offers methods for:
 *    - register all the states, and create evaluators for load/save/gather/scatter of states/parameters,
 *    - create evaluators for landice bcs (basal friction and lateral)
 *    - create evaluators for surface velocity and SMB diagnostistc
 *    - create evaluators for responses
 */
class StokesFOBase : public Albany::AbstractProblem {
public:

  // Some short names for types
  using IntrepidBasis    = Intrepid2::Basis<PHX::Device, RealType, RealType>;
  using IntrepidCubature = Intrepid2::Cubature<PHX::Device>;
  using FL  = Albany::FieldLocation;
  using FRT = Albany::FieldRankType;
  using FST = FieldScalarType;
  using IReq = InterpolationRequest;

  //! Return number of spatial dimensions
  int spatialDimension() const { return numDim; }

  //! Get boolean telling code if SDBCs are utilized
  bool useSDBCs() const {return use_sdbcs_; }

  //! Build the PDE instantiations, boundary conditions, and initial solution
  void buildProblem (Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
                     Albany::StateManager& stateMgr);

protected:
  StokesFOBase (const Teuchos::RCP<Teuchos::ParameterList>& params_,
                const Teuchos::RCP<Teuchos::ParameterList>& discParams_,
                const Teuchos::RCP<ParamLib>& paramLib_,
                const int numDim_);

  template <typename EvalT>
  void constructStokesFOBaseEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                        const Albany::MeshSpecsStruct& meshSpecs,
                                        Albany::StateManager& stateMgr,
                                        Albany::FieldManagerChoice fieldManagerChoice);

  template <typename EvalT>
  void constructStatesEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                  const Albany::MeshSpecsStruct& meshSpecs,
                                  Albany::StateManager& stateMgr,
                                  Albany::FieldManagerChoice fieldManagerChoice);

  template <typename EvalT>
  void constructVelocityEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                    const Albany::MeshSpecsStruct& meshSpecs,
                                    Albany::StateManager& stateMgr,
                                    Albany::FieldManagerChoice fieldManagerChoice);

  template <typename EvalT>
  void constructInterpolationEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0);

  template <typename EvalT>
  void constructSideUtilityFields (PHX::FieldManager<PHAL::AlbanyTraits>& fm0);

  template <typename EvalT>
  void constructBasalBCEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0);

  template <typename EvalT>
  void constructLateralBCEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0);

  template <typename EvalT>
  void constructSMBEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                               const Albany::MeshSpecsStruct& meshSpecs);

  template <typename EvalT> Teuchos::RCP<const PHX::FieldTag>
  constructStokesFOBaseResponsesEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                            const Albany::MeshSpecsStruct& meshSpecs,
                                            Albany::StateManager& stateMgr,
                                            Albany::FieldManagerChoice fieldManagerChoice,
                                            const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  virtual void constructDirichletEvaluators (const Albany::MeshSpecsStruct& /* meshSpecs */) {}
  virtual void constructNeumannEvaluators (const Teuchos::RCP<Albany::MeshSpecsStruct>& /* meshSpecs */) {}

  Teuchos::RCP<Teuchos::ParameterList>
  getStokesFOBaseProblemParameters () const;

  void setSingleFieldProperties (const std::string& fname,
                                 const FRT rank,
                                 const FST st = FST::Real);

  void parseInputFields ();

  std::string side_fname (const std::string& fname, const std::string& ss_name) const {
    return fname + "_" + ss_name;
  }
  std::string basal_fname (const std::string& fname) const {
    return side_fname(fname,basalSideName);
  }
  std::string surf_fname (const std::string& fname) const {
    return side_fname(fname,surfaceSideName);
  }

  // This method sets the properties of fields that need to be handled automatically (e.g., need interpolation evaluators)
  virtual void setFieldsProperties ();

  // Note: derived problems should override this function to add more requests. Their implementation should *most likely*
  //       include a call to the base class' implementation.
  virtual void setupEvaluatorRequests ();

  FST get_scalar_type (const std::string& fname);
  FRT get_field_rank (const std::string& fname);
  void add_dep (const std::string& fname, const std::string& dep_name);

  // Checks if a field with given name and layout is already computed
  // by some evaluator in the field manager.
  template<typename EvalT>
  bool is_available (const PHX::FieldManager<PHAL::AlbanyTraits>& fm,
                     const std::string& name,
                     const FRT rank, const FST st, const FL loc,
                     const Teuchos::RCP<Albany::Layouts>& dl);
  template<typename EvalT>
  bool is_available (const PHX::FieldManager<PHAL::AlbanyTraits>& fm,
                     const std::string& name, const FL loc,
                     const Teuchos::RCP<Albany::Layouts>& dl);

  // ------------------- Members ----------------- //

  // Topology, basis and cubature of cells
  Teuchos::RCP<shards::CellTopology>  cellType;
  Teuchos::RCP<IntrepidBasis>         cellBasis;
  Teuchos::RCP<IntrepidCubature>      cellCubature;

  // Topology, basis and cubature of side sets
  std::map<std::string,Teuchos::RCP<shards::CellTopology>>  sideType;
  std::map<std::string,Teuchos::RCP<IntrepidBasis>>         sideBasis;
  std::map<std::string,Teuchos::RCP<IntrepidCubature>>      sideCubature;

  //! Discretization parameters
  Teuchos::RCP<Teuchos::ParameterList> discParams;

  // Layouts
  Teuchos::RCP<Albany::Layouts> dl;

  // Parameter lists for LandIce-specific BCs
  std::map<LandIceBC,std::vector<Teuchos::RCP<Teuchos::ParameterList>>>  landice_bcs;

  // Surface side, where velocity diagnostics are computed (e.g., velocity mismatch)
  std::string surfaceSideName;

  // Basal side, where thickness-related diagnostics are computed (e.g., SMB)
  std::string basalSideName;

  // In these three, the entry [0] always refers to the velocity
  Teuchos::ArrayRCP<std::string> dof_names;
  Teuchos::ArrayRCP<int>         dof_offsets;
  Teuchos::ArrayRCP<std::string> resid_names;
  Teuchos::ArrayRCP<std::string> scatter_names;

  unsigned int numDim;
  unsigned int vecDimFO;

  /// Boolean marking whether SDBCs are used
  bool use_sdbcs_;

  // Whether to use corrected temperature in the viscosity
  bool viscosity_use_corrected_temperature;
  bool compute_dissipation;

  //Wether to compute rigid body modes
  bool computeConstantModes, computeRotationModes;

  template<typename T>
  using StrMap = std::map<std::string,T>;

  // Variables used to track properties of fields and parameters
  StrMap<bool>   is_input_field;
  StrMap<FL>     input_field_loc;
  StrMap<FRT>    field_rank;
  StrMap<FST>    field_scalar_type;

  StrMap<bool> is_dist;
  StrMap<bool> save_sensitivities;
  StrMap<std::string> dist_params_name_to_mesh_part;

  StrMap<StrMap<bool>>  is_ss_input_field;
  StrMap<StrMap<FL>>    ss_input_field_loc;

  StrMap<bool>  is_dist_param;
  StrMap<bool>  is_extruded_param;
  StrMap<int>   extruded_params_levels;

  // Track the utility evaluators that a field needs
  StrMap<std::map<IReq,bool>> build_interp_ev;
  StrMap<StrMap<std::map<IReq,bool>>> ss_build_interp_ev;

  // Track the utility evaluators needed by each side set
  StrMap<std::map<UtilityRequest,bool>>  ss_utils_needed;

  // This is used to automatically detect/establish the scalar type of some fields.
  StrMap<std::set<std::string>> field_deps;

  // Name of common variables (constructor provides defaults)
  std::string velocity_name;
  std::string body_force_name;
  std::string surface_height_name;
  std::string ice_thickness_name;
  std::string flux_divergence_name;
  std::string bed_topography_name;
  std::string temperature_name;
  std::string corrected_temperature_name;
  std::string flow_factor_name;
  std::string stiffening_factor_name;
  std::string effective_pressure_name;
  std::string basal_friction_name;
  std::string sliding_velocity_name;
  std::string vertically_averaged_velocity_name;

  //! Problem PL
  const Teuchos::RCP<Teuchos::ParameterList> params;

  template<typename T>
  std::string print_map_keys (const std::map<std::string,T>& map);
};

template <typename EvalT>
void StokesFOBase::
constructStokesFOBaseEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                 const Albany::MeshSpecsStruct& meshSpecs,
                                 Albany::StateManager& stateMgr,
                                 Albany::FieldManagerChoice fieldManagerChoice)
{
  // --- States/parameters --- //
  constructStatesEvaluators<EvalT> (fm0, meshSpecs, stateMgr, fieldManagerChoice);

  // --- Velocity evaluators --- //
  constructVelocityEvaluators<EvalT> (fm0, meshSpecs, stateMgr, fieldManagerChoice);

  // --- Lateral BC evaluators (if needed) --- //
  constructLateralBCEvaluators<EvalT> (fm0);

  // --- Basal BC evaluators (if needed) --- //
  constructBasalBCEvaluators<EvalT> (fm0);

  // --- SMB-related evaluators (if needed) --- //
  constructSMBEvaluators<EvalT> (fm0, meshSpecs);

  // --- Sides utility fields ---//
  constructSideUtilityFields<EvalT> (fm0);

  // --- Interpolation utilities for fields ---//
  // NOTE: this has to be done last, cause it uses information set by
  //       previous methods to detect if/how to build an interpolation
  //       evaluator
  constructInterpolationEvaluators<EvalT> (fm0);

  if (fieldManagerChoice == Albany::BUILD_RESID_FM) {
    // Require scattering of residuals
    for (const auto& s : scatter_names) {
      PHX::Tag<typename EvalT::ScalarT> res_tag(s, dl->dummy);
      fm0.requireField<EvalT>(res_tag);
    }
  }

  // For synthetic inverse problems, allow to save observations+noise
  // NOTE: do this only for EvalT=Residual, so any instantiation of scalar type works
  if (std::is_same<EvalT,PHAL::AlbanyTraits::Residual>::value) {
    auto& noise_pl = params->sublist("LandIce Noise");
    for (auto it : noise_pl) {
      const auto& pl = it.second.getValue<Teuchos::ParameterList>(0);
      auto ev = Teuchos::rcp(new PHAL::AddNoiseST<EvalT,PHAL::AlbanyTraits>(pl,dl));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }
}

template <typename EvalT>
void StokesFOBase::
constructStatesEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                           const Albany::MeshSpecsStruct& meshSpecs,
                           Albany::StateManager& stateMgr,
                           Albany::FieldManagerChoice fieldManagerChoice)
{
  const auto eval_name = PHX::print<EvalT>();

  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  // Temporary variables used numerous times below
  Albany::StateStruct::MeshFieldEntity entity;
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  // ---------------------------- Registering state variables ------------------------- //

  std::string stateName, fieldName;

  // Volume mesh requirements
  Teuchos::ParameterList& req_fields_info = discParams->sublist("Required Fields Info");
  unsigned int num_fields = req_fields_info.get<int>("Number Of Fields",0);

  std::string fieldType, fieldUsage, meshPart;
  Teuchos::RCP<PHX::DataLayout> state_dl;
  for (unsigned int ifield=0; ifield<num_fields; ++ifield) {
    Teuchos::ParameterList& thisFieldList = req_fields_info.sublist(Albany::strint("Field", ifield));

    // Get current state specs
    fieldName = thisFieldList.get<std::string>("Field Name");
    stateName = thisFieldList.get<std::string>("State Name", fieldName);
    fieldUsage = thisFieldList.get<std::string>("Field Usage","Input"); // WARNING: assuming Input if not specified

    if (fieldUsage == "Unused") {
      continue;
    }

    fieldType  = thisFieldList.get<std::string>("Field Type");

    is_dist_param.insert(std::pair<std::string,bool>(stateName, false)); //gets inserted only if not there.
    is_dist.insert(std::pair<std::string,bool>(stateName, false)); //gets inserted only if not there.

    meshPart = is_dist[stateName] ? dist_params_name_to_mesh_part[stateName] : "";

    auto loc = fieldType.find("Node")!=std::string::npos ? FL::Node : FL::Cell;
    auto rank = get_field_rank(stateName);

    // Get data layout
    if (rank == FRT::Scalar) {
      state_dl = loc == FL::Node ? dl->node_scalar : dl->cell_scalar2;
    } else if (rank == FRT::Vector) {
      state_dl = loc == FL::Node ? dl->node_vector : dl->cell_vector;
    } else if (rank == FRT::Gradient) {
      state_dl = loc == FL::Node ? dl->node_gradient : dl->cell_gradient;
    } else if (rank == FRT::Tensor) {
      state_dl = loc == FL::Node ? dl->node_tensor : dl->cell_tensor;
    }

    // Set entity for state struct
    if(loc == FL::Cell) {
      entity = Albany::StateStruct::ElemData;
    } else {
      if (is_dist[stateName]) {
        entity = Albany::StateStruct::NodalDistParameter;
      } else {
        entity = Albany::StateStruct::NodalDataToElemNode;
      }
    }

    if(is_dist[stateName] && save_sensitivities[stateName]) {
      p = stateMgr.registerStateVariable(stateName + "_sensitivity", state_dl, meshSpecs.ebName, true, &entity, meshPart);
    }
    p = stateMgr.registerStateVariable(stateName, state_dl, meshSpecs.ebName, true, &entity, meshPart);

    // Do we need to load/gather the state/parameter?
    if (is_dist[stateName]) {
      // A distributed field (likely a parameter): gather or scatter it (depending on whether is marked as input)
      if (is_input_field[stateName]) {
        // An input (not computed): gather it
        if (is_extruded_param[stateName]) {
          ev = evalUtils.constructGatherScalarExtruded2DNodalParameter(stateName,fieldName);
          fm0.template registerEvaluator<EvalT>(ev);
        } else {
          ev = evalUtils.constructGatherScalarNodalParameter(stateName,fieldName);
          fm0.template registerEvaluator<EvalT>(ev);
        }
      } else {
        // Not an input (must be computed). Scatter it.
        ev = evalUtils.constructScatterScalarNodalParameter(stateName,fieldName);
        fm0.template registerEvaluator<EvalT>(ev);
        // Only PHAL::AlbanyTraits::Residual evaluates something, others will have empty list of evaluated fields
        if (ev->evaluatedFields().size()>0) {
          fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
        }
      }
    } else {
      // Not a distributed field, that is, a simple state field.
      // Check if we need to load and/or save it.
      if (fieldUsage == "Output" || fieldUsage == "Input-Output") {
        // Only save fields in the residual FM (and not in state/response FM)
        if (fieldManagerChoice == Albany::BUILD_RESID_FM) {
          // An output: save it.
          p->set<bool>("Nodal State", loc==FL::Node);
          ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT,PHAL::AlbanyTraits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);

          // Only PHAL::AlbanyTraits::Residual evaluates something,
          // others will have empty list of evaluated fields
          if (ev->evaluatedFields().size()>0) {
            fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
          }
        }
      }

      if (fieldUsage == "Input" || fieldUsage == "Input-Output") {
        p->set<std::string>("Field Name", fieldName);
        // Note: input state fields should have scalar type (st) Real.
        //       However, to allow backward compatibility for some evaluators,
        //       they might have st ParamScalar, or even Scalar, so we allow
        //       loading them directly into an MDField with the correct st.
        auto st = get_scalar_type(stateName);
        if (st==FST::Scalar) {
          ev = Teuchos::rcp(new PHAL::LoadStateFieldST<EvalT,PHAL::AlbanyTraits>(*p));
        } else if (st==FST::ParamScalar) {
          ev = Teuchos::rcp(new PHAL::LoadStateFieldPST<EvalT,PHAL::AlbanyTraits>(*p));
        } else if (st==FST::MeshScalar) {
          ev = Teuchos::rcp(new PHAL::LoadStateFieldMST<EvalT,PHAL::AlbanyTraits>(*p));
        } else {
          ev = Teuchos::rcp(new PHAL::LoadStateFieldRT<EvalT,PHAL::AlbanyTraits>(*p));
        }
        fm0.template registerEvaluator<EvalT>(ev);
      }
    }
  }

  // Side set requirements
  Teuchos::Array<std::string> ss_names;
  if (discParams->sublist("Side Set Discretizations").isParameter("Side Sets")) {
    ss_names = discParams->sublist("Side Set Discretizations").get<Teuchos::Array<std::string>>("Side Sets");
  }
  for (unsigned int i=0; i<ss_names.size(); ++i) {
    const std::string& ss_name = ss_names[i];
    Teuchos::ParameterList& info = discParams->sublist("Side Set Discretizations").sublist(ss_name).sublist("Required Fields Info");
    num_fields = info.get<int>("Number Of Fields",0);
    int numLayers;

    const std::string& sideEBName = meshSpecs.sideSetMeshSpecs.at(ss_name)[0]->ebName;
    Teuchos::RCP<Albany::Layouts> ss_dl = dl->side_layouts.at(ss_name);
    for (unsigned int ifield=0; ifield<num_fields; ++ifield) {
      Teuchos::ParameterList& thisFieldList =  info.sublist(Albany::strint("Field", ifield));

      // Get current state specs
      fieldName = thisFieldList.get<std::string>("Field Name");
      stateName = thisFieldList.get<std::string>("State Name", fieldName);
      fieldName = fieldName + "_" + ss_name;
      fieldUsage = thisFieldList.get<std::string>("Field Usage","Input"); // WARNING: assuming Input if not specified

      if (fieldUsage == "Unused") {
        continue;
      }

      //meshPart = is_dist_param[stateName] ? dist_params_name_to_mesh_part[stateName] : "";
      meshPart = ""; // Distributed parameters are defined either on the whole volume mesh or on a whole side mesh. Either way, here we want "" as part (the whole mesh).

      fieldType  = thisFieldList.get<std::string>("Field Type");
      auto loc = fieldType.find("Node")!=std::string::npos ? FL::Node : FL::Cell;
      auto rank = get_field_rank(stateName);


      // Get data layout
      if (rank == FRT::Scalar) {
        state_dl = loc == FL::Node
                 ? ss_dl->node_scalar
                 : ss_dl->cell_scalar2;
      } else if (rank == FRT::Vector) {
        state_dl = loc == FL::Node
                 ? ss_dl->node_vector
                 : ss_dl->cell_vector;
      } else if (rank == FRT::Gradient) {
        state_dl = loc == FL::Node
                 ? ss_dl->node_gradient
                 : ss_dl->cell_gradient;
      } else if (rank == FRT::Tensor) {
        state_dl = loc == FL::Node
                 ? ss_dl->node_tensor
                 : ss_dl->cell_tensor;
      }

      // If layered, extend the layout
      if(fieldType.find("Layered")!=std::string::npos) {
        numLayers = thisFieldList.get<int>("Number Of Layers");
        state_dl = extrudeSideLayout(state_dl,numLayers);
      }

      // Set entity for state struct
      if(loc==FL::Cell) {
        entity = Albany::StateStruct::ElemData;
      } else {
        if (is_dist[stateName]) {
          entity = Albany::StateStruct::NodalDistParameter;
        } else {
          entity = Albany::StateStruct::NodalDataToElemNode;
        }
      }

      // Register the state
      p = stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, state_dl, sideEBName, true, &entity, meshPart);

      // Create load/save evaluator(s)
      // Note:
      //  - dist fields should not be loaded/gathered on the ss;
      //    instead, gather them in 3D, and project on the ss;
      //  - dist fields should not be saved on the ss if they are computed (i.e., not input fields),
      //    since they are not correct until scattered, which does not happen before projection.
      if ( !(is_dist[stateName] && !is_input_field[stateName]) &&
           (fieldUsage == "Output" || fieldUsage == "Input-Output")) {
        // Only save fields in the residual FM (and not in state/response FM)
        if (fieldManagerChoice == Albany::BUILD_RESID_FM) {
          // An output: save it.
          // SaveSideSetStateField takes the layout from dl, using FRT and FL to determine it.
          // It does so, in order to do J*v if v is a Gradient (covariant), where J is the 2x3
          // matrix of the tangent vectors
          p->set("Field Rank",rank);
          p->set("Field Location",loc);
          ev = Teuchos::rcp(new PHAL::SaveSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p,ss_dl));
          fm0.template registerEvaluator<EvalT>(ev);

          // Only PHAL::AlbanyTraits::Residual evaluates something, others will have empty list of evaluated fields
          if (ev->evaluatedFields().size()>0) {
            fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
          }
        }
      }

      if (!is_dist[stateName] && (fieldUsage == "Input" || fieldUsage == "Input-Output")) {
        p->set<std::string>("Field Name", fieldName);
        // Note: input state fields should have scalar type (st) Real.
        //       However, to allow backward compatibility for some evaluators,
        //       they might have st ParamScalar, or even Scalar, so we allow
        //       loading them directly into an MDField with the correct st.
        auto st = get_scalar_type(stateName);
        if (st==FST::Scalar) {
          ev = Teuchos::rcp(new PHAL::LoadSideSetStateFieldST<EvalT,PHAL::AlbanyTraits>(*p));
        } else if (st==FST::ParamScalar) {
          ev = Teuchos::rcp(new PHAL::LoadSideSetStateFieldPST<EvalT,PHAL::AlbanyTraits>(*p));
        } else if (st==FST::MeshScalar) {
          ev = Teuchos::rcp(new PHAL::LoadSideSetStateFieldMST<EvalT,PHAL::AlbanyTraits>(*p));
        } else {
          ev = Teuchos::rcp(new PHAL::LoadSideSetStateFieldRT<EvalT,PHAL::AlbanyTraits>(*p));
        }
        fm0.template registerEvaluator<EvalT>(ev);
      }
    }
  }
}

template <typename EvalT>
void StokesFOBase::
constructInterpolationEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  std::map<FST,Teuchos::RCP<const Albany::EvaluatorUtilsBase<PHAL::AlbanyTraits>>> utils_map;
  utils_map[FST::Scalar]      = Teuchos::rcpFromRef(evalUtils.getSTUtils());
  utils_map[FST::ParamScalar] = Teuchos::rcpFromRef(evalUtils.getPSTUtils());
  utils_map[FST::MeshScalar]  = Teuchos::rcpFromRef(evalUtils.getMSTUtils());
  utils_map[FST::Real]        = Teuchos::rcpFromRef(evalUtils.getRTUtils());

  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;

  // Loop on all input fields
  for (auto& it : build_interp_ev) {

    // Get the field name
    const std::string& fname = it.first;

    // If there's no information about this field, we assume it is not needed, so we skip it.
    // If it WAS indeed needed, Phalanx DAG will miss a node, and an exception will be thrown.
    const auto st = get_scalar_type(fname);

    TEUCHOS_TEST_FOR_EXCEPTION (utils_map.find(st)==utils_map.end(), std::runtime_error,
                                "Error! Evaluators utils for scalar type '" + e2str(st) + "' not found.\n");
    const auto& utils = *utils_map.at(st);

    // Check whether we can use memoization for this field. Criteria:
    //  - need to have memoization enabled
    //  - cannot be a solution-dependent field
    //  - cannot be a distributed parameter
    //  - if param scalar, params must not depend on solution
    //  - if mesh scalar, mesh must not depend on solution/parameters
    // Note: if a field depends on a distributed parameter (and not on the solution), it would pass these tests,
    //       but memoization would be wrong. Therefore, we actually enable memoization if there are NO dist params.
    //       A better choice would be to track dependencies, but we cannot do this easily. We should probably
    //       have our own Albany::FieldManager (and Albany::Evaluator), which checks that all evaluators that
    //       try to use memoization actually can do it (i.e., they do not depend on fields that cannot be memoized).
    //
    // TODO: We can now track dependencies but we need to specify which MDField changes based on whether
    //       the dist param changes
    //
//    bool useMemoization = enableMemoizer && (is_dist_param.size()==0) && st!=FieldScalarType::Scalar;
//    if (st==FieldScalarType::ParamScalar) {
//      useMemoization &= !Albany::params_depend_on_solution();
//    } else if (st==FieldScalarType::MeshScalar) {
//      useMemoization &= !Albany::mesh_depends_on_solution() && !Albany::mesh_depends_on_parameters();
//    }

    // Get the needs of this field
    auto& needs = it.second;
    const auto rank = get_field_rank(fname);

    // For dofs, we can get a faster interpolation, knowing the offset
    auto dof_it = std::find(dof_names.begin(),dof_names.end(),fname);
    int offset = dof_it==dof_names.end() ? -1 : dof_offsets[std::distance(dof_names.begin(),dof_it)];

    if (needs[IReq::QP_VAL]) {
      switch (rank) {
        case FRT::Scalar:
          ev = utils.constructDOFInterpolationEvaluator(fname, offset);
          break;
        case FRT::Vector:
          ev = utils.constructDOFVecInterpolationEvaluator(fname, offset);
          break;
        case FRT::Tensor:
          ev = utils.constructDOFTensorInterpolationEvaluator(fname, offset);
          break;
        default:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
              "Error! Cannot interpolate to the quad points a '" + e2str(rank) + "' field.\n");
      }
      fm0.template registerEvaluator<EvalT> (ev);
    }

    if (needs[IReq::GRAD_QP_VAL]) {
      switch (rank) {
        case FRT::Scalar:
          ev = utils.constructDOFGradInterpolationEvaluator(fname, offset);
          break;
        case FRT::Vector:
          ev = utils.constructDOFVecGradInterpolationEvaluator(fname, offset);
          break;
        case FRT::Tensor:
          ev = utils.constructDOFTensorGradInterpolationEvaluator(fname, offset);
          break;
        default:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                "Error! Cannot interpolate to the quad points the gradient of a '" + e2str(rank) + "' field.\n");
      }
      fm0.template registerEvaluator<EvalT> (ev);
    }

    if (needs[IReq::CELL_VAL]) {
      if (is_available<EvalT>(fm0,fname,FL::Node,dl)) {
        ev = utils.constructBarycenterEvaluator (fname, cellBasis, rank);
        fm0.template registerEvaluator<EvalT> (ev);
      } else if (is_available<EvalT>(fm0,fname,FL::QuadPoint,dl)) {
        const std::string interpolationType = "Cell Average";
        ev = utils.constructP0InterpolationEvaluator (fname, interpolationType, FL::QuadPoint, rank);
        fm0.template registerEvaluator<EvalT> (ev);
      }
    }
  }

  // Loop on all side sets
  for (auto& it_outer : ss_build_interp_ev) {
    const std::string& ss_name = it_outer.first;
    const std::string eval_name = PHX::print<EvalT>();

    // Loop on all fields
    for (auto& it : it_outer.second) {

      // Get field name (with and without side name)
      const std::string fname = it.first;
      const std::string fname_side = fname + "_" + ss_name;

      // Get the needs of this field
      auto& needs = it.second;

      // Get location, rank, and scalar type of the field.
      // If we are missing some information about this field, we assume it is not needed, so we skip it.
      // If it WAS indeed needed, Phalanx DAG will miss a node, and an exception will be thrown.
      // Note: we don't know if the st of fname is set or the st of fname_side is set,
      //       so use op| to get the strongest (if one is not set, get_scalar_type returns Real).
      const auto st     = get_scalar_type(fname) | get_scalar_type(fname_side);

      // The st of the field if it did undergo certain interpolation (e.g., gradient or cell average)
      // that pollute its scalar type with that of the mesh.
      const auto st_mst = st | FST::MeshScalar;

      const auto rank = get_field_rank(fname);

      // Shortcut to the util function is_available, for this field (volume and side version)
      auto is_available_2d = [&](const FL loc) -> bool {
        return is_available<EvalT>(fm0,fname_side,rank,st,loc,dl->side_layouts.at(ss_name));
      };
      auto is_available_3d = [&](const FL loc) -> bool {
        return is_available<EvalT>(fm0,fname,rank,st,loc,dl);
      };
      // Same as the above, but with st_mst instead of st
      auto is_available_2d_mst = [&](const FL loc) -> bool {
        return is_available<EvalT>(fm0,fname_side,rank,st_mst,loc,dl->side_layouts.at(ss_name));
      };
      auto is_available_3d_mst = [&](const FL loc) -> bool {
        return is_available<EvalT>(fm0,fname,rank,st_mst,loc,dl);
      };

      // Check whether we can use memoization for this field. Criteria:
      //  - need to have memoization enabled
      //  - cannot be a solution-dependent field
      //  - cannot be a distributed parameter
      //  - if param scalar, params must not depend on solution
      //  - if mesh scalar, mesh must not depend on solution/parameters
      // Note: if a field depends on a distributed parameter (and not on the solution), it would pass these tests,
      //       but memoization would be wrong. Therefore, we actually enable memoization if there are NO dist params.
      //       A better choice would be to track dependencies, but we cannot do this easily. We should probably
      //       have our own Albany::FieldManager (and Albany::Evaluator), which checks that all evaluators that
      //       try to use memoization actually can do it (i.e., they do not depend on fields that cannot be memoized).
      //
      // TODO: We can now track dependencies but we need to specify which MDField changes based on whether
      //       the dist param changes
      //
//      bool useMemoization = enableMemoizer && (is_dist_param.size()==0) && st!=FieldScalarType::Scalar;
//      if (st==FieldScalarType::ParamScalar) {
//        useMemoization &= !Albany::params_depend_on_solution();
//      } else if (st==FieldScalarType::MeshScalar) {
//        useMemoization &= !Albany::mesh_depends_on_solution() && !Albany::mesh_depends_on_parameters();
//      }

      // Get the right evaluator utils for this field.
      TEUCHOS_TEST_FOR_EXCEPTION (utils_map.find(st)==utils_map.end(), std::runtime_error,
            "Error! Evaluators utils for scalar type " + e2str(st) + " not found on ss: " + ss_name + ".\n");
      TEUCHOS_TEST_FOR_EXCEPTION (utils_map.find(st_mst)==utils_map.end(), std::runtime_error,
            "Error! Evaluators utils for scalar type " + e2str(st_mst) + " not found on ss: " + ss_name + ".\n");

      // Utils with this field's st, and with st or-ed with MeshScalar,
      // which is needed if the field underwent transformations that polluted
      // its st with the mesh st (e.g., gradient or cell average)
      const auto& utils     = *utils_map.at(st);
      const auto& utils_mst = *utils_map.at(st | FST::MeshScalar);

      // Project to the side only if it is requested and it is NOT already available on the side.
      // Note: do this first so that is_available_2d can be used for other interpolation checks below.
      if ( needs[IReq::CELL_TO_SIDE] ) {
        if (!is_available_2d(FL::Node) && is_available_3d(FL::Node)) {
          // Project from cell to side
          const std::string layout = e2str(FL::Node) + " " + e2str(rank);
          ev = utils.constructDOFCellToSideEvaluator(fname, ss_name, layout, cellType, fname_side);
          fm0.template registerEvaluator<EvalT> (ev);
        }
        // For loc==Cell, if the cell field was computed via CellAverage, the st should be st_mst
        if (!is_available_2d(FL::Cell) && is_available_3d(FL::Cell)) {
          // Project from cell to side
          const std::string layout = e2str(FL::Cell) + " " + e2str(rank);
          ev = utils.constructDOFCellToSideEvaluator(fname, ss_name, layout, cellType, fname_side);
          fm0.template registerEvaluator<EvalT> (ev);
        } else if (!is_available_2d_mst(FL::Cell) && is_available_3d_mst(FL::Cell)) {
          // Project from cell to side
          const std::string layout = e2str(FL::Cell) + " " + e2str(rank);
          ev = utils_mst.constructDOFCellToSideEvaluator(fname, ss_name, layout, cellType, fname_side);
          fm0.template registerEvaluator<EvalT> (ev);
        }
      }

      if (needs[IReq::QP_VAL] && is_available_2d(FL::Node) && !is_available_2d(FL::QuadPoint)) {
        TEUCHOS_TEST_FOR_EXCEPTION (rank!=FRT::Scalar && rank!=FRT::Vector, std::logic_error,
            "Error! Interpolation on side only available for scalar and vector fields.\n");
        if (rank==FRT::Scalar) {
          ev = utils.constructDOFInterpolationSideEvaluator (fname_side, ss_name);
        } else {
          ev = utils.constructDOFVecInterpolationSideEvaluator (fname_side, ss_name);
        }
        fm0.template registerEvaluator<EvalT> (ev);
      }

      if (needs[IReq::GRAD_QP_VAL] && is_available_2d(FL::Node)) {
        // Put "_gradient" before the ss name, so you can save the field via input file,
        // since side states field names are BLAH+ss_name, where BLAH is the name
        // specified in the input file.
        const std::string& grad_name_side = fname + "_gradient_" + ss_name;
        if (rank==FRT::Scalar) {
          ev = utils.constructDOFGradInterpolationSideEvaluator (fname_side, ss_name, grad_name_side);
        } else if (rank==FRT::Vector) {
          ev = utils.constructDOFVecGradInterpolationSideEvaluator (fname_side, ss_name);
        } else {
          TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
              "Error! Gradient interpolation on side only available for scalar and vector fields.\n");
        }
        fm0.template registerEvaluator<EvalT> (ev);
      }

      if (needs[IReq::CELL_VAL]) {
        // Interpolate field at Side from Node/QuadPoints values
        // CAREFUL: If the rank is Gradient, then the input's scalar typee is st_mst
        //           For Scalar/Vector/Tensor quantities, the field st is correct.
        //           Also, skip if somehow the Cell field is already computed,
        //           perhaps by an ad-hoc physics evaluator.
        if (rank==FRT::Gradient) {
          if (is_available_2d_mst(FL::QuadPoint) && !is_available_2d_mst(FL::Cell)) {
            ev = utils_mst.constructCellAverageSideEvaluator (ss_name, fname_side, FL::QuadPoint, rank);
            fm0.template registerEvaluator<EvalT> (ev);
          }
        } else if (!is_available_2d(FL::Cell)) {
          if (is_available_2d(FL::QuadPoint)) {
            ev = utils.constructCellAverageSideEvaluator (ss_name, fname_side, FL::QuadPoint, rank);
            fm0.template registerEvaluator<EvalT> (ev);
          } else if (is_available_2d(FL::Node)) {
            ev = utils.constructCellAverageSideEvaluator (ss_name, fname_side, FL::Node, rank);
            fm0.template registerEvaluator<EvalT> (ev);
          }
        }
      }
    }
  }
}

template <typename EvalT>
void StokesFOBase::
constructSideUtilityFields (PHX::FieldManager<PHAL::AlbanyTraits>& fm0)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;

  for (auto& it : ss_utils_needed) {
    const std::string& ss_name = it.first;

    //---- Compute side basis functions
    if (it.second[UtilityRequest::BFS] || it.second[UtilityRequest::NORMALS]) {
      // BF, GradBF, w_measure, Tangents, Metric, Metric Det, Inverse Metric
      ev = evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, sideBasis[ss_name], sideCubature[ss_name],
                                                                 ss_name, it.second[UtilityRequest::NORMALS]);
      fm0.template registerEvaluator<EvalT> (ev);
    }

    if (it.second[UtilityRequest::QP_COORDS]) {
      // QP coordinates
      ev = evalUtils.constructMapToPhysicalFrameSideEvaluator(cellType, sideCubature[ss_name], ss_name);
      fm0.template registerEvaluator<EvalT> (ev);

      // Baricenter coordinate
      ev = evalUtils.getMSTUtils().constructBarycenterSideEvaluator(ss_name,Albany::coord_vec_name + "_" + ss_name, sideBasis[ss_name], FRT::Gradient);
      fm0.template registerEvaluator<EvalT> (ev);
    }

    // If any of the above was true, we need coordinates of vertices on the side
    if (it.second[UtilityRequest::BFS] || it.second[UtilityRequest::QP_COORDS] || it.second[UtilityRequest::NORMALS]) {
      ev = evalUtils.getMSTUtils().constructDOFCellToSideEvaluator(Albany::coord_vec_name,ss_name,"Vertex Vector",cellType,Albany::coord_vec_name + "_" + ss_name);
      fm0.template registerEvaluator<EvalT> (ev);
    }
  }
}

template <typename EvalT>
void StokesFOBase::
constructVelocityEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                             const Albany::MeshSpecsStruct& meshSpecs,
                             Albany::StateManager& stateMgr,
                             Albany::FieldManagerChoice fieldManagerChoice)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  std::string param_name;

  // ------------------- Interpolations and utilities ------------------ //

  // Map to physical frame
  ev = evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  // Compute basis funcitons
  ev = evalUtils.constructComputeBasisFunctionsEvaluator(cellType, cellBasis, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  // Get coordinate of cell baricenter
  ev = evalUtils.getMSTUtils().constructCellAverageEvaluator(Albany::coord_vec_name, FL::QuadPoint, FRT::Gradient);
  fm0.template registerEvaluator<EvalT> (ev);

  // -------------------------------- LandIce evaluators ------------------------- //

  // --- FO Stokes Stress --- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Stokes Stress"));

  //Input
  p->set<std::string>("Velocity QP Variable Name", velocity_name);
  p->set<std::string>("Velocity Gradient QP Variable Name", velocity_name + " Gradient");
  p->set<std::string>("Viscosity QP Variable Name", "LandIce Viscosity");
  p->set<std::string>("Surface Height QP Name", surface_height_name);
  p->set<std::string>("Coordinate Vector Name", Albany::coord_vec_name);
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
  p->set<Teuchos::ParameterList*>("Physical Parameter List", &params->sublist("LandIce Physical Parameters"));

  //Output
  p->set<std::string>("Stress Variable Name", "Stress Tensor");

  ev = createEvaluatorWithOneScalarType<StokesFOStress,EvalT>(p,dl,get_scalar_type(surface_height_name));
  fm0.template registerEvaluator<EvalT>(ev);

  // --- FO Stokes Resid --- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Stokes Resid"));

  //Input
  p->set<std::string>("Weighted BF Variable Name", Albany::weighted_bf_name);
  p->set<std::string>("Weighted Gradient BF Variable Name", Albany::weighted_grad_bf_name);
  p->set<std::string>("Velocity QP Variable Name", velocity_name);
  p->set<std::string>("Velocity Gradient QP Variable Name", velocity_name + " Gradient");
  p->set<std::string>("Body Force Variable Name", body_force_name);
  p->set<std::string>("Viscosity QP Variable Name", "LandIce Viscosity");
  p->set<std::string>("Coordinate Vector Name", Albany::coord_vec_name);
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("Equation Set"));

  //Output
  p->set<std::string>("Residual Variable Name", resid_names[0]);

  ev = Teuchos::rcp(new StokesFOResid<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Shared Parameter for Continuation: Glen's Law Homotopy Parameter ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Homotopy Parameter"));

  param_name = ParamEnumName::GLHomotopyParam;
  p->set<std::string>("Parameter Name", param_name);
  p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
  p->set<const Teuchos::ParameterList*>("Parameters List", &params->sublist("Parameters"));
  p->set<double>("Default Nominal Value", params->sublist("LandIce Viscosity").get<double>(param_name,-1.0));

  Teuchos::RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::GLHomotopy>> ptr_gl_homotopy;
  ptr_gl_homotopy = Teuchos::rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::GLHomotopy>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ptr_gl_homotopy);

  //--- LandIce Flow Rate ---//
  auto& visc_pl = params->sublist("LandIce Viscosity");
  if (visc_pl.isParameter("Flow Rate Type") &&
      !is_available<EvalT>(fm0,flow_factor_name,FL::Cell,dl)) {
    if((visc_pl.get<std::string>("Flow Rate Type","Uniform") == "From File") ||
       (visc_pl.get<std::string>("Flow Rate Type","Uniform") == "From CISM")) {
      // The field *should* already be specified as an 'Elem Scalar' required field in the mesh.
    } else {
      p = Teuchos::rcp(new Teuchos::ParameterList("LandIce FlowRate"));

      //Input
      const auto& temp_name = viscosity_use_corrected_temperature
                            ? corrected_temperature_name
                            : temperature_name;

      p->set<std::string>("Temperature Variable Name", temp_name);
      p->set<Teuchos::ParameterList*>("Parameter List", &visc_pl);

      //Output
      p->set<std::string>("Flow Rate Variable Name", flow_factor_name);

      ev = createEvaluatorWithOneScalarType<FlowRate,EvalT>(p,dl,get_scalar_type(temp_name));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  //--- LandIce viscosity ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Viscosity"));

  //Input
  p->set<std::string>("Coordinate Vector Variable Name", Albany::coord_vec_name);
  p->set<std::string>("Velocity QP Variable Name", velocity_name);
  p->set<std::string>("Velocity Gradient QP Variable Name", velocity_name + " Gradient");
  std::string visc_temp_name = viscosity_use_corrected_temperature
                             ? corrected_temperature_name
                             : temperature_name;
  p->set<std::string>("Temperature Variable Name", visc_temp_name);
  p->set<std::string>("Ice Softness Variable Name", flow_factor_name);
  p->set<std::string>("Stiffening Factor QP Name", stiffening_factor_name);
  p->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("LandIce Viscosity"));
  p->set<std::string>("Continuation Parameter Name",ParamEnumName::GLHomotopyParam);

  //Output
  p->set<std::string>("Viscosity QP Variable Name", "LandIce Viscosity");
  p->set<std::string>("EpsilonSq QP Variable Name", "LandIce EpsilonSq");

  // The st of T for viscosity is complicated: you need to get the st of the T used (temp or corrected temp),
  // and consider that you do Nodes->Cell interp, which introduces MeshScalar type in the result.
  FST temp_st = get_scalar_type(visc_temp_name);
  if (!is_available<EvalT>(fm0,visc_temp_name,FL::Node,dl)) {
    // Temperature is not available at nodes (for some reason). We'll have to
    // do P0 interpolation as a CellAverage, which divides by w_measure, hence
    // polluting the output ST with MeshScalarT.
    temp_st |= FST::MeshScalar;
  }
  ev = createEvaluatorWithTwoScalarTypes<ViscosityFO,EvalT>(p,dl,FST::Scalar,temp_st);
  fm0.template registerEvaluator<EvalT>(ev);

  // --- Print LandIce Dissipation ---
  if(compute_dissipation) {

    // LandIce Dissipation
    p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Dissipation"));

    //Input
    p->set<std::string>("Viscosity QP Variable Name", "LandIce Viscosity");
    p->set<std::string>("EpsilonSq QP Variable Name", "LandIce EpsilonSq");
    p->set<Teuchos::ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));

    //Output
    p->set<std::string>("Dissipation QP Variable Name", "LandIce Dissipation");

    ev = Teuchos::rcp(new Dissipation<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    ev = evalUtils.getPSTUtils().constructCellAverageEvaluator("LandIce Dissipation",FL::QuadPoint, FRT::Scalar);
    fm0.template registerEvaluator<EvalT> (ev);

    // Saving the dissipation heat in the output mesh
    std::string stateName = "dissipation_heat";
    auto entity = Albany::StateStruct::ElemData;
    p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, meshSpecs.ebName, true, &entity);
    p->set<std::string>("Field Name", "LandIce Dissipation");
    p->set<std::string>("Weights Name",Albany::weights_name);
    p->set("Weights Layout", dl->qp_scalar);
    p->set("Field Layout", dl->cell_scalar2);
    p->set< Teuchos::RCP<PHX::DataLayout> >("Dummy Data Layout",dl->dummy);
    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    if (fieldManagerChoice == Albany::BUILD_RESID_FM) {
      // Only PHAL::AlbanyTraits::Residual evaluates something
      if (ev->evaluatedFields().size()>0) {
        // Require save friction heat
        fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
      }
    }
  }

  // Saving the stress tensor in the output mesh
  if(params->get<bool>("Print Stress Tensor", false))
  {
    // Interpolate stress tensor, from qps to a single cell scalar
    ev = evalUtils.constructCellAverageEvaluator("Stress Tensor", FL::QuadPoint, FRT::Tensor);
    fm0.template registerEvaluator<EvalT> (ev);

    // Save stress tensor (if needed)
    std::string stateName = "Stress Tensor";
    auto entity = Albany::StateStruct::ElemData;
    p = stateMgr.registerStateVariable(stateName, dl->cell_tensor, meshSpecs.ebName, true, &entity);
    p->set< Teuchos::RCP<PHX::DataLayout> >("State Field Layout",dl->cell_tensor);
    p->set<std::string>("Field Name", "Stress Tensor");
    p->set< Teuchos::RCP<PHX::DataLayout> >("Dummy Data Layout",dl->dummy);
    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    if (fieldManagerChoice == Albany::BUILD_RESID_FM) {
      // Only PHAL::AlbanyTraits::Residual evaluates something
      if (ev->evaluatedFields().size()>0) {
        // Require save friction heat
        fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
      }
    }
  }

#ifdef CISM_HAS_LANDICE
  //--- LandIce surface gradient from CISM ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Surface Gradient"));

  //Input
  p->set<std::string>("CISM Surface Height Gradient X Variable Name", "xgrad_surface_height");
  p->set<std::string>("CISM Surface Height Gradient Y Variable Name", "ygrad_surface_height");
  p->set<std::string>("BF Variable Name", Albany::bf_name);

  //Output
  p->set<std::string>("Surface Height Gradient QP Variable Name", "CISM Surface Height Gradient");
  ev = Teuchos::rcp(new CismSurfaceGradFO<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);
#endif

  //--- Body Force ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Body Force"));

  //Input
  p->set<std::string>("LandIce Viscosity QP Variable Name", "LandIce Viscosity");
#ifdef CISM_HAS_LANDICE
  p->set<std::string>("Surface Height Gradient QP Variable Name", "CISM Surface Height Gradient");
#endif
  p->set<std::string>("Coordinate Vector Variable Name", Albany::coord_vec_name);
  p->set<std::string>("Surface Height Gradient Name", surface_height_name + " Gradient");
  p->set<std::string>("Surface Height Name", surface_height_name);
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("Body Force"));
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
  p->set<Teuchos::ParameterList*>("Physical Parameter List", &params->sublist("LandIce Physical Parameters"));

  //Output
  p->set<std::string>("Body Force Variable Name", body_force_name);

  ev = createEvaluatorWithOneScalarType<StokesFOBodyForce,EvalT>(p,dl,get_scalar_type(surface_height_name));
  fm0.template registerEvaluator<EvalT>(ev);

  // ---------- Add time as a Sacado-ized parameter (only if specified) ------- //
  bool isTimeAParameter = false;
  if (params->isParameter("Use Time Parameter")) isTimeAParameter = params->get<bool>("Use Time Parameter");
  if (isTimeAParameter) {
    p = Teuchos::rcp(new Teuchos::ParameterList("Time"));
    p->set<Teuchos::RCP<PHX::DataLayout>>("Workset Scalar Data Layout", dl->workset_scalar);
    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);
    p->set<bool>("Disable Transient", true);
    p->set<std::string>("Time Name", "Time");
    p->set<std::string>("Delta Time Name", "Delta Time");
    ev = Teuchos::rcp(new Time<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerStateVariable("Time", dl->workset_scalar, dl->dummy, meshSpecs.ebName, "scalar", 0.0, true);
    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }
}

template <typename EvalT>
void StokesFOBase::constructBasalBCEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  std::string param_name;

  for (auto pl : landice_bcs[LandIceBC::BasalFriction]) {
    const std::string& ssName = pl->get<std::string>("Side Set Name");

    auto dl_side = dl->side_layouts.at(ssName);
    auto is_available_2d = [&](const std::string& fname, const FL loc) -> bool {
      const std::string fname_side = side_fname(fname,ssName);
      auto st = get_scalar_type(fname);
      auto rank = get_field_rank(fname);
      return is_available<EvalT>(fm0,fname_side,rank,st,loc,dl->side_layouts.at(ssName));
    };

    // We may have more than 1 basal side set. The layout of all the side fields is the
    // same, so we need to differentiate them by name (just like we do for the basis functions already).

    std::string velocity_side_name           = side_fname(velocity_name, ssName);
    std::string sliding_velocity_side_name   = side_fname(sliding_velocity_name, ssName);
    std::string beta_side_name               = side_fname("beta", ssName);
    std::string ice_thickness_side_name      = side_fname(ice_thickness_name, ssName);
    std::string ice_overburden_side_name     = side_fname("ice_overburden", ssName);
    std::string effective_pressure_side_name = side_fname(effective_pressure_name, ssName);
    std::string bed_roughness_side_name      = side_fname("bed_roughness", ssName);
    std::string bed_topography_side_name     = side_fname(bed_topography_name, ssName);
    std::string flow_factor_side_name        = side_fname(flow_factor_name, ssName);

    // -------------------------------- LandIce evaluators ------------------------- //

    // --- Basal Residual --- //
    p = Teuchos::rcp(new Teuchos::ParameterList("Stokes Basal Residual"));

    //Input
    p->set<std::string>("BF Side Name", side_fname(Albany::bf_name,ssName));
    p->set<std::string>("Weighted Measure Name", side_fname(Albany::weighted_measure_name,ssName));
    p->set<std::string>("Basal Friction Coefficient Side QP Variable Name", beta_side_name);
    p->set<std::string>("Velocity Side QP Variable Name", velocity_side_name);
    p->set<std::string>("Side Set Name", ssName);
    p->set<std::string>("Side Normal Name", side_fname(Albany::normal_name, ssName));
    p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type", cellType);

    //Output
    p->set<std::string>("Residual Variable Name", resid_names[0]);

    ev = Teuchos::rcp(new StokesFOBasalResid<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    //--- Sliding velocity calculation at nodes ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Velocity Norm"));

    // Input
    p->set<std::string>("Field Name",velocity_side_name);
    p->set<std::string>("Field Layout","Cell Side Node Vector");
    p->set<std::string>("Side Set Name", ssName);
    p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("LandIce Field Norm"));

    // Output
    p->set<std::string>("Field Norm Name",sliding_velocity_side_name);

    if (!is_available_2d(sliding_velocity_name,FL::Node)) {
      ev = Teuchos::rcp(new PHAL::FieldFrobeniusNorm<EvalT,PHAL::AlbanyTraits>(*p,dl_side));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    //--- Sliding velocity calculation ---//
    if (!is_available_2d(sliding_velocity_name,FL::QuadPoint)) {
      p->set<std::string>("Field Layout","Cell Side QuadPoint Vector");
      ev = Teuchos::rcp(new PHAL::FieldFrobeniusNorm<EvalT,PHAL::AlbanyTraits>(*p,dl_side));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    //--- Ice Overburden (QPs) ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Ice IceOverburden"));

    // Input
    p->set<bool>("Nodal",false);
    p->set<std::string>("Side Set Name", ssName);
    p->set<std::string>("Ice Thickness Variable Name", ice_thickness_side_name);
    p->set<Teuchos::ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));

    // Output
    p->set<std::string>("Ice Overburden Variable Name", ice_overburden_side_name);

    if (!is_available_2d("ice_overburden",FL::QuadPoint)) {
      ev = Teuchos::rcp(new IceOverburden<EvalT,PHAL::AlbanyTraits>(*p,dl_side));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    //--- Ice Overburden (Nodes) ---//
    if (!is_available_2d("ice_overburden",FL::Node)) {
      p->set<bool>("Nodal",true);
      ev = Teuchos::rcp(new IceOverburden<EvalT,PHAL::AlbanyTraits>(*p,dl_side));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    // If we are given an effective pressure field or if a subclass sets up an evaluator to compute it,
    // we don't need a surrogate model for it
    if (!is_input_field[effective_pressure_name] && !is_ss_input_field[ssName][effective_pressure_name]) {
      p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Effective Pressure Surrogate"));

      // Input
      p->set<bool>("Nodal",false);
      p->set<std::string>("Side Set Name", ssName);
      p->set<std::string>("Ice Overburden Variable Name", ice_overburden_side_name);

      // Output
      p->set<std::string>("Effective Pressure Variable Name", effective_pressure_side_name);

      if (!is_available_2d(effective_pressure_name,FL::QuadPoint)) {
        //--- Effective pressure surrogate (QPs) ---//
        ev = Teuchos::rcp(new EffectivePressure<EvalT,PHAL::AlbanyTraits,true>(*p,dl_side));
        fm0.template registerEvaluator<EvalT>(ev);
      }
      if (!is_available_2d(effective_pressure_name,FL::Node)) {
        //--- Effective pressure surrogate (QPs) ---//
        p->set<bool>("Nodal",true);
        ev = Teuchos::rcp(new EffectivePressure<EvalT,PHAL::AlbanyTraits,true>(*p,dl_side));
        fm0.template registerEvaluator<EvalT>(ev);
      }

      //--- Shared Parameter for basal friction coefficient: alpha ---//
      p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: alpha"));

      param_name = "Hydraulic-Over-Hydrostatic Potential Ratio";
      p->set<std::string>("Parameter Name", param_name);
      p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
      p->set<const Teuchos::ParameterList*>("Parameters List", &params->sublist("Parameters"));

      //TODO Why is this parameter looked for in the Basal Friction List?
      double nominalValue = pl->sublist("Basal Friction Coefficient").isParameter(param_name) ?
                               pl->sublist("Basal Friction Coefficient").get<double>(param_name) : -1.0;
      p->set<double>("Default Nominal Value", nominalValue);

      Teuchos::RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Alpha>> ptr_alpha;
      ptr_alpha = Teuchos::rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Alpha>(*p,dl));
      fm0.template registerEvaluator<EvalT>(ptr_alpha);
    }

    //--- Shared Parameter for basal friction coefficient: lambda ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: lambda"));

    param_name = "Bed Roughness";
    p->set<std::string>("Parameter Name", param_name);
    p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
    p->set<const Teuchos::ParameterList*>("Parameters List", &params->sublist("Parameters"));
    p->set<double>("Default Nominal Value", pl->sublist("Basal Friction Coefficient").get<double>(param_name,-1.0));

    Teuchos::RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Lambda>> ptr_lambda;
    ptr_lambda = Teuchos::rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Lambda>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ptr_lambda);

    //--- Shared Parameter for basal friction coefficient: muPowerLaw ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: mu"));

    param_name = ParamEnumName::Mu;
    p->set<std::string>("Parameter Name", param_name);
    p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
    p->set<const Teuchos::ParameterList*>("Parameters List", &params->sublist("Parameters"));
    p->set<double>("Default Nominal Value", pl->sublist("Basal Friction Coefficient").get<double>(param_name,-1.0));

    Teuchos::RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Mu>> ptr_mu;
    ptr_mu = Teuchos::rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Mu>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ptr_mu);

    //--- Shared Parameter for basal friction coefficient: power ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: power"));

    param_name = "Power Exponent";
    p->set<std::string>("Parameter Name", param_name);
    p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
    p->set<const Teuchos::ParameterList*>("Parameters List", &params->sublist("Parameters"));
    Teuchos::ParameterList beta_list = pl->sublist("Basal Friction Coefficient");
    const auto type = util::upper_case(beta_list.get<std::string>("Type"));
    double default_val = (type == "FIELD") ? 1.0 : beta_list.get<double>(param_name, -1.0);
    p->set<double>("Default Nominal Value", default_val);

    Teuchos::RCP<PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Power>> ptr_power;
    ptr_power = Teuchos::rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits,ParamEnum,ParamEnum::Power>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ptr_power);

    //--- LandIce basal friction coefficient ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Basal Friction Coefficient"));

    //Input
    p->set<std::string>("Sliding Velocity Variable Name", sliding_velocity_side_name);
    p->set<std::string>("BF Variable Name", side_fname(Albany::bf_name, ssName));
    p->set<std::string>("Effective Pressure QP Variable Name", effective_pressure_side_name);
    p->set<std::string>("Ice Softness Variable Name", flow_factor_side_name);
    p->set<std::string>("Bed Roughness Variable Name", bed_roughness_side_name);
    p->set<std::string>("Side Set Name", ssName);
    p->set<std::string>("Coordinate Vector Variable Name", side_fname(Albany::coord_vec_name, ssName));
    p->set<Teuchos::ParameterList*>("Parameter List", &pl->sublist("Basal Friction Coefficient"));
    p->set<Teuchos::ParameterList*>("Physical Parameter List", &params->sublist("LandIce Physical Parameters"));
    p->set<Teuchos::ParameterList*>("Viscosity Parameter List", &params->sublist("LandIce Viscosity"));
    p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
    p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
    p->set<std::string>("Bed Topography Variable Name", bed_topography_side_name);
    p->set<std::string>("Effective Pressure Variable Name", effective_pressure_side_name);
    p->set<std::string>("Ice Thickness Variable Name", ice_thickness_side_name);
    p->set<bool>("Is Thickness A Parameter",is_dist_param[ice_thickness_name]);
    p->set<Teuchos::RCP<std::map<std::string,bool>>>("Dist Param Query Map",Teuchos::rcpFromRef(is_dist_param));

    //Output
    p->set<std::string>("Basal Friction Coefficient Variable Name", beta_side_name);

    std::string bft = util::upper_case(pl->sublist("Basal Friction Coefficient").get<std::string>("Type"));

    auto N_st = get_scalar_type(effective_pressure_name);
    auto A_st = get_scalar_type(flow_factor_name);
    ev = createEvaluatorWithThreeScalarTypes<BasalFrictionCoefficient,EvalT>(p,dl_side,N_st,FST::Scalar,A_st);
    fm0.template registerEvaluator<EvalT>(ev);

    //--- LandIce basal friction coefficient at nodes ---//
    p->set<bool>("Nodal",true);
    ev = createEvaluatorWithThreeScalarTypes<BasalFrictionCoefficient,EvalT>(p,dl_side,N_st,FST::Scalar,A_st);
    fm0.template registerEvaluator<EvalT>(ev);
  }
}

template <typename EvalT>
void StokesFOBase::constructLateralBCEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  for (auto pl : landice_bcs[LandIceBC::Lateral]) {
    const std::string& ssName = pl->get<std::string>("Side Set Name");

    // We may have more than 1 lateral side set. The layout of all the side fields is the
    // same, so we need to differentiate them by name (just like we do for the basis functions already).

    std::string ice_thickness_side_name  = side_fname(ice_thickness_name,ssName);
    std::string surface_height_side_name = side_fname(surface_height_name,ssName);

    // -------------------------------- LandIce evaluators ------------------------- //

    // Lateral residual
    p = Teuchos::rcp( new Teuchos::ParameterList("Lateral Residual") );

    // Input
    p->set<std::string>("Ice Thickness Variable Name", ice_thickness_side_name);
    p->set<std::string>("Ice Surface Elevation Variable Name", surface_height_side_name);
    p->set<std::string>("Coordinate Vector Variable Name", side_fname(Albany::coord_vec_name, ssName));
    p->set<std::string>("BF Side Name", side_fname(Albany::bf_name, ssName));
    p->set<std::string>("Weighted Measure Name", side_fname(Albany::weighted_measure_name, ssName));
    p->set<std::string>("Side Normal Name", side_fname(Albany::normal_name, ssName));
    p->set<std::string>("Side Set Name", ssName);
    p->set<Teuchos::RCP<shards::CellTopology>>("Cell Type", cellType);
    p->set<Teuchos::ParameterList*>("Lateral BC Parameters",pl.get());
    p->set<Teuchos::ParameterList*>("Physical Parameters",&params->sublist("LandIce Physical Parameters"));
    p->set<Teuchos::ParameterList*>("Stereographic Map",&params->sublist("Stereographic Map"));

    // Output
    p->set<std::string>("Residual Variable Name", resid_names[0]);

    ev = createEvaluatorWithOneScalarType<StokesFOLateralResid,EvalT>(p,dl,get_scalar_type(ice_thickness_name));
    fm0.template registerEvaluator<EvalT>(ev);
  }
}

template <typename EvalT>
void StokesFOBase::constructSMBEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                           const Albany::MeshSpecsStruct& meshSpecs)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  // Evaluators needed for thickness-related diagnostics (e.g., SMB)
  if (!isInvalid(basalSideName)) {
    auto dl_side = dl->side_layouts.at(basalSideName);

    std::string basalSideNamePlanar = basalSideName + "_planar";

    {
      //---- Compute side basis functions
      auto ss_util_needed = ss_utils_needed[basalSideName];
      if (ss_util_needed[UtilityRequest::BFS] || ss_util_needed[UtilityRequest::NORMALS]) {
        // BF, GradBF, w_measure, Tangents, Metric, Metric Det, Inverse Metric
        ev = evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, sideBasis[basalSideName], sideCubature[basalSideName],
            basalSideName, false, true);
        fm0.template registerEvaluator<EvalT> (ev);
      }
    }

    // We may have more than 1 basal side set. 'basalSideName' should be the union of all of them.
    // However, some of the fields used here, may be used also to compute quantities defined on
    // only some of the sub-sidesets of 'basalSideName'. The layout of all the side fields is the
    // same, so we need to differentiate them by name (just like we do for the basis functions already).

    std::string velocity_side_name                     = basal_fname(velocity_name);
    std::string ice_thickness_side_name                = basal_fname(ice_thickness_name);
    std::string ice_thickness_side_name_planar         = side_fname(ice_thickness_name, basalSideNamePlanar);
    std::string surface_height_side_name               = basal_fname(surface_height_name);
    std::string apparent_mass_balance_side_name        = basal_fname("apparent_mass_balance");
    std::string apparent_mass_balance_RMS_side_name    = basal_fname("apparent_mass_balance_RMS");
    std::string stiffening_factor_side_name            = basal_fname(stiffening_factor_name);
    std::string effective_pressure_side_name           = basal_fname(effective_pressure_name);
    std::string vertically_averaged_velocity_side_name = basal_fname(vertically_averaged_velocity_name);
    std::string bed_roughness_side_name                = basal_fname("bed_roughness");

    // ------------------- Interpolations and utilities ------------------ //

    // if (is_dist_param[ice_thickness_name])
    // {
    //   //---- Restrict ice thickness from cell-based to cell-side-based
    //   ev = evalUtils.getPSTUtils().constructDOFCellToSideEvaluator(ice_thickness_name,basalSideName,"Node Scalar",cellType,ice_thickness_side_name);
    //   fm0.template registerEvaluator<EvalT> (ev);
    // }

    // -------------------------------- LandIce evaluators ------------------------- //

    {

      std::map<FieldScalarType,Teuchos::RCP<const Albany::EvaluatorUtilsBase<PHAL::AlbanyTraits>>> utils_map;
      utils_map[FieldScalarType::Scalar]      = Teuchos::rcpFromRef(evalUtils.getSTUtils());
      utils_map[FieldScalarType::ParamScalar] = Teuchos::rcpFromRef(evalUtils.getPSTUtils());
      utils_map[FieldScalarType::MeshScalar]  = Teuchos::rcpFromRef(evalUtils.getMSTUtils());
      utils_map[FieldScalarType::Real]        = Teuchos::rcpFromRef(evalUtils.getRTUtils());

      // If there's no information about this field, we assume it is not needed, so we skip it.
      // If it WAS indeed needed, Phalanx DAG will miss a node, and an exception will be thrown.
      const FieldScalarType st = get_scalar_type(ice_thickness_name);

      // Get the right evaluator utils for this field.
      // const FieldScalarType st = field_scalar_type.at(ice_thickness_name);
      const auto& utils = *utils_map.at(st);

      ev = utils.constructDOFGradInterpolationSideEvaluator (ice_thickness_side_name, basalSideName, true);
      fm0.template registerEvaluator<EvalT> (ev);
    }

    // Vertically averaged velocity
    p = Teuchos::rcp(new Teuchos::ParameterList("Gather Averaged Velocity"));

    p->set<std::string>("Contracted Solution Name", vertically_averaged_velocity_side_name);
    p->set<std::string>("Mesh Part", "basalside");
    p->set<int>("Solution Offset", dof_offsets[0]);
    p->set<std::string>("Side Set Name", basalSideName);
    p->set<bool>("Is Vector", true);
    p->set<std::string>("Contraction Operator", "Vertical Average");

    p->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",Teuchos::rcp(new CellTopologyData(meshSpecs.ctd)));

    ev = Teuchos::rcp(new GatherVerticallyContractedSolution<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    if(!params->isSublist("LandIce Flux Divergence") ||
       !params->sublist("LandIce Flux Divergence").get<bool>("Flux Divergence Is Part Of Solution")){
      // Flux divergence
      p = Teuchos::rcp(new Teuchos::ParameterList("Flux Divergence"));

      //Input
      p->set<std::string>("Averaged Velocity Side QP Variable Name", vertically_averaged_velocity_side_name);
      p->set<std::string>("Averaged Velocity Side QP Divergence Name", vertically_averaged_velocity_side_name + " Divergence");
      p->set<std::string>("Thickness Side QP Variable Name", ice_thickness_side_name);
      p->set<std::string>("Thickness Gradient Name", ice_thickness_side_name + " Planar Gradient");
      p->set<std::string>("Side Tangents Name", side_fname(Albany::tangents_name,basalSideNamePlanar));

      p->set<std::string>("Field Name",  "flux_divergence_basalside");
      p->set<std::string>("Side Set Name", basalSideName);

      ev = createEvaluatorWithOneScalarType<FluxDiv,EvalT>(p,dl_side,get_scalar_type(ice_thickness_name));
      fm0.template registerEvaluator<EvalT>(ev);

      // --- 2D divergence of Averaged Velocity ---- //
      p = Teuchos::rcp(new Teuchos::ParameterList("DOF Div Interpolation Side Averaged Velocity"));

      // Input
      p->set<std::string>("Variable Name", vertically_averaged_velocity_side_name);
      p->set<std::string>("Gradient BF Name", side_fname(Albany::grad_bf_name,basalSideNamePlanar));
      p->set<std::string>("Tangents Name", side_fname(Albany::tangents_name,basalSideNamePlanar));
      p->set<std::string>("Side Set Name",basalSideName);

      // Output (assumes same Name as input)
      p->set<std::string>("Divergence Variable Name", vertically_averaged_velocity_side_name  + " Divergence");

      ev = Teuchos::rcp(new DOFDivInterpolationSide<EvalT,PHAL::AlbanyTraits>(*p,dl_side));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }
}

template<typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
StokesFOBase::
constructStokesFOBaseResponsesEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                          const Albany::MeshSpecsStruct& meshSpecs,
                                          Albany::StateManager& stateMgr,
                                          Albany::FieldManagerChoice fieldManagerChoice,
                                          const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {

    // --- SurfaceVelocity-related evaluators (if needed) --- //
    Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList("Param List"));

    // Figure out if observed surface velocity RMS is scalar (if present at all)
    if (!isInvalid(surfaceSideName)) {
      if (is_ss_input_field[surfaceSideName]["observed_surface_velocity_RMS"]) {
        if (get_field_rank("observed_surface_velocity_RMS")==FRT::Scalar) {
          paramList->set<bool>("Scalar RMS",true);
        }
      }
    }

    // ----------------------- Responses --------------------- //

    paramList->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
    paramList->set<Teuchos::ParameterList>("LandIce Physical Parameters List", params->sublist("LandIce Physical Parameters"));
    paramList->set<Teuchos::RCP<std::map<std::string, int>>> ("Extruded Params Levels", Teuchos::rcpFromRef(extruded_params_levels));
    paramList->set<std::string>("Coordinate Vector Side Variable Name", basal_fname(Albany::coord_vec_name));
    paramList->set<std::string>("Basal Friction Coefficient Name", basal_friction_name);
    paramList->set<std::string>("Stiffening Factor Gradient Name",basal_fname(stiffening_factor_name + "_gradient"));
    paramList->set<std::string>("Stiffening Factor Name", basal_fname(stiffening_factor_name));
    paramList->set<std::string>("Thickness Side Variable Name",basal_fname(ice_thickness_name));
    paramList->set<std::string>("Bed Topography Side Variable Name",basal_fname(bed_topography_name));
    paramList->set<std::string>("Surface Velocity Side QP Variable Name",surf_fname(velocity_name));
    paramList->set<std::string>("Averaged Vertical Velocity Side Variable Name",basal_fname(vertically_averaged_velocity_name));
    paramList->set<std::string>("Observed Surface Velocity Side QP Variable Name",surf_fname("observed_surface_velocity"));
    paramList->set<std::string>("Observed Surface Velocity RMS Side QP Variable Name",surf_fname("observed_surface_velocity_RMS"));
    paramList->set<std::string>("Flux Divergence Side QP Variable Name",basal_fname("flux_divergence"));
    paramList->set<std::string>("Thickness RMS Side QP Variable Name",basal_fname("observed_ice_thickness_RMS"));
    paramList->set<std::string>("Observed Thickness Side QP Variable Name",basal_fname("observed_ice_thickness"));
    paramList->set<std::string>("SMB Side QP Variable Name",basal_fname("apparent_mass_balance"));
    paramList->set<std::string>("SMB RMS Side QP Variable Name",basal_fname("apparent_mass_balance_RMS"));
    paramList->set<std::string>("Thickness Gradient Name", basal_fname(ice_thickness_name) + " Planar Gradient");
    paramList->set<std::string>("Thickness Side QP Variable Name",basal_fname(ice_thickness_name));
    paramList->set<std::string>("Basal Side Name", basalSideName);
    paramList->set<std::string>("Weighted Measure Basal Name",basal_fname(Albany::weighted_measure_name));
    paramList->set<std::string>("Weighted Measure Surface Name",surf_fname(Albany::weighted_measure_name));
    paramList->set<std::string>("Metric 2D Name",basal_fname(Albany::metric_name));
    paramList->set<std::string>("Metric Basal Name",basal_fname(Albany::metric_name));
    paramList->set<std::string>("Metric Surface Name",surf_fname(Albany::metric_name));
    paramList->set<std::string>("Basal Side Tangents Name",basal_fname(Albany::tangents_name));
    paramList->set<std::string>("Weighted Measure 2D Name",side_fname(Albany::weighted_measure_name, basalSideName + "_planar"));
    paramList->set<std::string>("Inverse Metric Basal Name",basal_fname(Albany::metric_inv_name));
    paramList->set<std::string>("Surface Side Name", surfaceSideName);
    paramList->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",Teuchos::rcp(new CellTopologyData(meshSpecs.ctd)));
    paramList->set<std::vector<Teuchos::RCP<Teuchos::ParameterList>>*>("Basal Regularization Params",&landice_bcs[LandIceBC::BasalFriction]);
    paramList->set<FST>("Ice Thickness Scalar Type",get_scalar_type(ice_thickness_name));

    ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, paramList, stateMgr);
  }

  return Teuchos::null;
}

template<typename T>
std::string StokesFOBase::
print_map_keys (const std::map<std::string,T>& map) {
  std::string s;
  for (auto it : map) {
    s += " ";
    s += it.first;
  }
  return s;
}

template<typename EvalT>
bool StokesFOBase::
is_available (const PHX::FieldManager<PHAL::AlbanyTraits>& fm,
                   const std::string& name,
                   const FRT rank, const FST st, const FL loc,
                   const Teuchos::RCP<Albany::Layouts>& layouts) {
  using lt_ptr = Teuchos::RCP<PHX::DataLayout>;

  // Helper map
  std::map<FRT,std::map<FL,lt_ptr>> map;

  map[FRT::Scalar][FL::Node]      = layouts->node_scalar;
  map[FRT::Scalar][FL::Cell]      = layouts->cell_scalar2;
  map[FRT::Scalar][FL::QuadPoint] = layouts->qp_scalar;

  map[FRT::Vector][FL::Node]      = layouts->node_vector;
  map[FRT::Vector][FL::Cell]      = layouts->cell_vector;
  map[FRT::Vector][FL::QuadPoint] = layouts->qp_vector;

  map[FRT::Gradient][FL::Node]      = layouts->node_gradient;
  map[FRT::Gradient][FL::Cell]      = layouts->cell_gradient;
  map[FRT::Gradient][FL::QuadPoint] = layouts->qp_gradient;

  map[FRT::Tensor][FL::Node]      = layouts->node_tensor;
  map[FRT::Tensor][FL::Cell]      = layouts->cell_tensor;
  map[FRT::Tensor][FL::QuadPoint] = layouts->qp_tensor;

  auto lt = map.at(rank).at(loc);

  auto tag = createTag<EvalT>(name,st,lt);

  const auto& dag = fm.getDagManager<EvalT>();

  const auto& field_to_eval = dag.queryRegisteredFields();
  auto search = std::find_if(field_to_eval.begin(),
                             field_to_eval.end(),
                             [&] (const auto& tag_identifier)
                             {return (tag->identifier() == tag_identifier.first);});

  return search!=field_to_eval.end();
}

template<typename EvalT>
bool StokesFOBase::
is_available (const PHX::FieldManager<PHAL::AlbanyTraits>& fm,
              const std::string& name, const FL loc,
              const Teuchos::RCP<Albany::Layouts>& layouts) {
  auto st = get_scalar_type(name);
  auto rank = get_field_rank(name);

  return is_available<EvalT>(fm,name,rank,st,loc,layouts);
}

} // namespace LandIce

#endif // LANDICE_STOKES_FO_BASE_HPP
