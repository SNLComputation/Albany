//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LINEARELASTICITYPROBLEM_HPP
#define LINEARELASTICITYPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"
#include "Albany_BCUtils.hpp"
#include "ATO_OptimizationProblem.hpp"
#include "ATO_Mixture.hpp"
#include "ATO_Utils.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"


namespace Albany {

  /*!
   * \brief Abstract interface for representing a 2-D finite element
   * problem.
   */
  class LinearElasticityProblem : 
    public ATO::OptimizationProblem ,
    public virtual Albany::AbstractProblem {
  public:
  
    //! Default constructor
    LinearElasticityProblem(
		      const Teuchos::RCP<Teuchos::ParameterList>& params_,
		      const Teuchos::RCP<ParamLib>& paramLib_,
		      const int numDim_);

    //! Destructor
    virtual ~LinearElasticityProblem();

    //! Return number of spatial dimensions
    virtual int spatialDimension() const { return numDim; }

    //! Build the PDE instantiations, boundary conditions, and initial solution
    virtual void buildProblem(
      Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
      StateManager& stateMgr);

    // Build evaluators
    virtual Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
    buildEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecsStruct& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    //! Each problem must generate it's list of valid parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

    void getAllocatedStates(
      Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device> > > > oldState_,
      Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device> > > > newState_) const;

  private:

    //! Private to prohibit copying
    LinearElasticityProblem(const LinearElasticityProblem&);
    
    //! Private to prohibit copying
    LinearElasticityProblem& operator=(const LinearElasticityProblem&);

  public:

    //! Main problem setup routine. Not directly called, but indirectly by following functions
    template <typename EvalT> 
    Teuchos::RCP<const PHX::FieldTag>
    constructEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecsStruct& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList);

    void constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);
    void constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);


  protected:

    int numDim;

    Teuchos::RCP<Albany::Layouts> dl;

    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device> > > > oldState;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device> > > > newState;

//    Teuchos::RCP<QCAD::MaterialDatabase> material_db_;

  };

}

#include "Albany_SolutionAverageResponseFunction.hpp"
#include "Albany_SolutionTwoNormResponseFunction.hpp"
#include "Albany_SolutionMaxValueResponseFunction.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_EvaluatorUtils.hpp"

#include "PHAL_GatherScalarNodalParameter.hpp"
#include "PHAL_Source.hpp"
#include "Strain.hpp"
#include "ATO_Stress.hpp"
#include "ATO_TopologyFieldWeighting.hpp"
#include "ATO_TopologyWeighting.hpp"
#include "PHAL_SaveStateField.hpp"
#include "ElasticityResid.hpp"

#include "Time.hpp"

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::LinearElasticityProblem::constructEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fieldManagerChoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
   using Teuchos::RCP;
   using Teuchos::rcp;
   using Teuchos::ParameterList;
   using PHX::DataLayout;
   using PHX::MDALayout;
   using std::vector;
   using PHAL::AlbanyTraits;

  // get the name of the current element block
   std::string elementBlockName = meshSpecs.ebName;

   RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));
   RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> >
     intrepidBasis = Albany::getIntrepid2Basis(meshSpecs.ctd);

   const int numNodes = intrepidBasis->getCardinality();
   const int worksetSize = meshSpecs.worksetSize;

   Intrepid2::DefaultCubatureFactory cubFactory;
   RCP <Intrepid2::Cubature<PHX::Device> > cubature = cubFactory.create<PHX::Device, RealType, RealType>(*cellType, meshSpecs.cubatureDegree);

   const int numDim = cubature->getDimension();
   const int numQPts = cubature->getNumPoints();
   const int numVertices = cellType->getNodeCount();

   *out << "Field Dimensions: Workset=" << worksetSize 
        << ", Vertices= " << numVertices
        << ", Nodes= " << numNodes
        << ", QuadPts= " << numQPts
        << ", Dim= " << numDim << std::endl;


   // Construct standard FEM evaluators with standard field names                              
   dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim));
   TEUCHOS_TEST_FOR_EXCEPTION(dl->vectorAndGradientLayoutsAreEquivalent==false, std::logic_error,
                              "Data Layout Usage in Mechanics problems assume vecDim = numDim");

   Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
   ATO::Utils<EvalT, PHAL::AlbanyTraits> atoUtils(dl);



   std::string stressName("Stress");
   std::string strainName("Strain");


   Teuchos::ArrayRCP<std::string> dof_names(1);
   dof_names[0] = "Displacement";
   Teuchos::ArrayRCP<std::string> resid_names(1);
   resid_names[0] = dof_names[0]+" Residual";

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0]));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0]));

   fm0.template registerEvaluator<EvalT>
       (evalUtils.constructGatherSolutionEvaluator_noTransient(/*is_vector_dof=*/ true, dof_names));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructScatterResidualEvaluator(/*is_vector_dof=*/ true, resid_names));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructGatherCoordinateVectorEvaluator());

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature));

   fm0.template registerEvaluator<EvalT>
     (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));

   // Temporary variable used numerous times below
   Teuchos::RCP<PHX::Evaluator<AlbanyTraits> > ev;


  { // Time
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Time Name", "Time");
    p->set<std::string>("Delta Time Name", "Delta Time");
    p->set< RCP<DataLayout> >("Workset Scalar Data Layout", dl->workset_scalar);
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    p->set<bool>("Disable Transient", true);

    ev = rcp(new LCM::Time<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerStateVariable("Time",dl->workset_scalar, dl->dummy, 
                                       elementBlockName, "scalar", 0.0, true);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Strain
    RCP<ParameterList> p = rcp(new ParameterList(strainName));

    //Input
    p->set<std::string>("Gradient QP Variable Name", "Displacement Gradient");

    //Output
    p->set<std::string>("Strain Name", strainName);

    ev = rcp(new LCM::Strain<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    // state the strain in the state manager for ATO
    p = stateMgr.registerStateVariable(strainName, dl->qp_tensor, dl->dummy, 
                                       elementBlockName, "scalar", 0.0, false, false);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    //if(some input stuff)
    atoUtils.SaveCellStateField(fm0, stateMgr, strainName, elementBlockName, dl->qp_tensor, numDim);
  }


  { // Linear elasticity stress
    RCP<ParameterList> p = rcp(new ParameterList(stressName));

    //Input
    p->set<std::string>("Strain Name", strainName);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    if( params->isType<int>("Add Cell Problem Forcing") )
      p->set<int>("Cell Forcing Column",params->get<int>("Add Cell Problem Forcing") );

    // check for multiple element block specs
    Teuchos::ParameterList& configParams = params->sublist("Configuration");

    if( configParams.isSublist("Element Blocks") ){
      Teuchos::ParameterList& blocksParams = configParams.sublist("Element Blocks");
      int nblocks = blocksParams.get<int>("Number of Element Blocks");
      bool blockFound = false;
      for(int ib=0; ib<nblocks; ib++){
        Teuchos::ParameterList& blockParams = blocksParams.sublist(Albany::strint("Element Block", ib));
        std::string blockName = blockParams.get<std::string>("Name");
        if( blockName != elementBlockName ) continue;
        blockFound = true;

        // user can specify a material or a mixture
        if( blockParams.isSublist("Material") ){
          // parse material
          Teuchos::ParameterList& materialParams = blockParams.sublist("Material",false);
          if( materialParams.isSublist("Homogenized Constants") ){
            Teuchos::ParameterList& homogParams = p->sublist("Homogenized Constants",false);
            homogParams.setParameters(materialParams.sublist("Homogenized Constants",true));
            p->set<Albany::StateManager*>("State Manager", &stateMgr);
            p->set<Teuchos::RCP<Albany::Layouts> >("Data Layout", dl);
          } else {
            p->set<double>("Elastic Modulus", materialParams.get<double>("Elastic Modulus"));
            p->set<double>("Poissons Ratio",  materialParams.get<double>("Poissons Ratio"));
          }
          //Output
          p->set<std::string>("Stress Name", stressName);

          ev = rcp(new ATO::Stress<EvalT,AlbanyTraits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);
      
          // state the strain in the state manager so for ATO
          p = stateMgr.registerStateVariable(stressName,dl->qp_tensor, dl->dummy, 
                                             elementBlockName, "scalar", 0.0, false, false);
          ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);
      
          //if(some input stuff)
          atoUtils.SaveCellStateField(fm0, stateMgr, stressName, elementBlockName, dl->qp_tensor, numDim);

        } else
        if( blockParams.isSublist("Mixture") ){
          // parse mixture
          Teuchos::ParameterList& mixtureParams = blockParams.sublist("Mixture",false);
          int nmats = mixtureParams.get<int>("Number of Materials");

          //-- create individual materials --//
          for(int imat=0; imat<nmats; imat++){
            RCP<ParameterList> pmat = rcp(new ParameterList(*p));
            Teuchos::ParameterList& materialParams = mixtureParams.sublist(Albany::strint("Material", imat));
            if( materialParams.isSublist("Homogenized Constants") ){
              Teuchos::ParameterList& homogParams = pmat->sublist("Homogenized Constants",false);
              homogParams.setParameters(materialParams.sublist("Homogenized Constants",true));
              pmat->set<Albany::StateManager*>("State Manager", &stateMgr);
              pmat->set<Teuchos::RCP<Albany::Layouts> >("Data Layout", dl);
            } else {
              pmat->set<double>("Elastic Modulus", materialParams.get<double>("Elastic Modulus"));
              pmat->set<double>("Poissons Ratio",  materialParams.get<double>("Poissons Ratio"));
            }
            //Output
            std::string outName = Albany::strint(stressName, imat);
            pmat->set<std::string>("Stress Name", outName);
  
            ev = rcp(new ATO::Stress<EvalT,AlbanyTraits>(*pmat));
            fm0.template registerEvaluator<EvalT>(ev);
        
            // state the strain in the state manager so for ATO
            pmat = stateMgr.registerStateVariable(outName,dl->qp_tensor, dl->dummy, 
                                                  elementBlockName, "scalar", 0.0, false, false);
            ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*pmat));
            fm0.template registerEvaluator<EvalT>(ev);
        
            //if(some input stuff)
            atoUtils.SaveCellStateField(fm0, stateMgr, outName, elementBlockName, dl->qp_tensor, numDim);
          }

          //-- create mixture --//
          TEUCHOS_TEST_FOR_EXCEPTION( !mixtureParams.isSublist("Mixed Fields"), std::logic_error,
                                  "'Mixture' requested but no 'Fields' defined"  << std::endl <<
                                  "Add 'Fields' list");
          {
            Teuchos::ParameterList& fieldsParams = mixtureParams.sublist("Mixed Fields",false);
            int nfields = fieldsParams.get<int>("Number of Mixed Fields");


            //-- create individual mixture field evaluators --//
            for(int ifield=0; ifield<nfields; ifield++){
              Teuchos::ParameterList& fieldParams = fieldsParams.sublist(Albany::strint("Mixed Field", ifield));
              std::string fieldName = fieldParams.get<std::string>("Field Name");

              RCP<ParameterList> p = rcp(new ParameterList(fieldName + " Mixed Field"));

              std::string fieldLayout = fieldParams.get<std::string>("Field Layout");
              p->set<std::string>("Field Layout", fieldLayout);

              std::string mixtureRule = fieldParams.get<std::string>("Rule Type");

              // currently only SIMP-type mixture is implemented
              Teuchos::ParameterList& simpParams = fieldParams.sublist(mixtureRule);
 
              Teuchos::RCP<ATO::TopologyArray> 
                topologyArray = params->get<Teuchos::RCP<ATO::TopologyArray> >("Topologies");
              p->set<Teuchos::RCP<ATO::TopologyArray> > ("Topologies", topologyArray);

              // topology and function indices
              p->set<Teuchos::Array<int> >("Topology Indices", 
                                           simpParams.get<Teuchos::Array<int> >("Topology Indices"));
              p->set<Teuchos::Array<int> >("Function Indices", 
                                           simpParams.get<Teuchos::Array<int> >("Function Indices"));

              // constituent var names
              Teuchos::Array<int> matIndices = simpParams.get<Teuchos::Array<int> >("Material Indices");
              int nMats = matIndices.size();
              Teuchos::Array<int> topoIndices = simpParams.get<Teuchos::Array<int> >("Topology Indices");
              int nTopos = topoIndices.size();
              TEUCHOS_TEST_FOR_EXCEPTION(nMats != nTopos+1, std::logic_error, std::endl <<
                                        "For SIMP Mixture, 'Materials' list must be 1 longer than 'Topologies' list"
                                        << std::endl);
              Teuchos::Array<std::string> constituentNames(nMats);
              for(int imat=0; imat<nmats; imat++){
                std::string constituentName = Albany::strint(fieldName, matIndices[imat]);
                constituentNames[imat] = constituentName;
              }
              p->set<Teuchos::Array<std::string> >("Constituent Variable Names", constituentNames);
              
              // mixture var name
              p->set<std::string>("Mixture Variable Name",fieldName);

              // basis functions
              p->set<std::string>("BF Name", "BF");
    
              ev = rcp(new ATO::Mixture<EvalT,AlbanyTraits>(*p,dl));
              fm0.template registerEvaluator<EvalT>(ev);
            }
          }
        } else
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                                  "'Material' or 'Mixture' not specified for '" 
                                  << elementBlockName << "'");
      }
      TEUCHOS_TEST_FOR_EXCEPTION(!blockFound, std::logic_error,
                                 "Material definition for block named '" << elementBlockName << "' not found");
    } else {

      if( params->isSublist("Homogenized Constants") ){
        Teuchos::ParameterList& homogParams = p->sublist("Homogenized Constants",false);
        homogParams.setParameters(params->sublist("Homogenized Constants",true));
        p->set<Albany::StateManager*>("State Manager", &stateMgr);
        p->set<Teuchos::RCP<Albany::Layouts> >("Data Layout", dl);
      } else {
        p->set<double>("Elastic Modulus", params->get<double>("Elastic Modulus"));
        p->set<double>("Poissons Ratio",  params->get<double>("Poissons Ratio"));
      }
    
      //Output
      p->set<std::string>("Stress Name", stressName);

      ev = rcp(new ATO::Stress<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
  
      // state the strain in the state manager so for ATO
      p = stateMgr.registerStateVariable(stressName,dl->qp_tensor, dl->dummy, 
                                         elementBlockName, "scalar", 0.0, false, false);
      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
  
      //if(some input stuff)
      atoUtils.SaveCellStateField(fm0, stateMgr, stressName, elementBlockName, dl->qp_tensor, numDim);
    }

  }
  /*******************************************************************************/
  /** Begin topology weighting ***************************************************/
  /*******************************************************************************/
  if(params->isType<Teuchos::RCP<ATO::TopologyArray> >("Topologies"))
  {
    Teuchos::RCP<ATO::TopologyArray> 
      topologyArray = params->get<Teuchos::RCP<ATO::TopologyArray> >("Topologies");

    Teuchos::ParameterList& wfParams = params->sublist("Apply Topology Weight Functions");
    int nfields = wfParams.get<int>("Number of Fields");
    for(int ifield=0; ifield<nfields; ifield++){
      Teuchos::ParameterList& fieldParams = wfParams.sublist(Albany::strint("Field", ifield));

      int topoIndex  = fieldParams.get<int>("Topology Index");
      std::string fieldName  = fieldParams.get<std::string>("Name");
      std::string layoutName = fieldParams.get<std::string>("Layout");
      int functionIndex      = fieldParams.get<int>("Function Index");

      Teuchos::RCP<PHX::DataLayout> layout;
      if( layoutName == "QP Scalar" ) layout = dl->qp_scalar;
      else
      if( layoutName == "QP Vector" ) layout = dl->qp_vector;
      else
      if( layoutName == "QP Tensor" ) layout = dl->qp_tensor;

      Teuchos::RCP<ATO::Topology> topology = (*topologyArray)[topoIndex];

      // Get distributed parameter
      if( topology->getEntityType() == "Distributed Parameter" ){
        RCP<ParameterList> p = rcp(new ParameterList("Distributed Parameter"));
        p->set<std::string>("Parameter Name", topology->getName());
        ev = rcp(new PHAL::GatherScalarNodalParameter<EvalT,AlbanyTraits>(*p, dl) );
        fm0.template registerEvaluator<EvalT>(ev);
      }
  
      RCP<ParameterList> p = rcp(new ParameterList("TopologyWeighting"));

      p->set<Teuchos::RCP<ATO::Topology> >("Topology",topology);

      p->set<std::string>("BF Name", "BF");
      p->set<std::string>("Unweighted Variable Name", fieldName);
      p->set<std::string>("Weighted Variable Name", fieldName+"_Weighted");
      p->set<std::string>("Variable Layout", layoutName);
      p->set<int>("Function Index", functionIndex);

      if( topology->getEntityType() == "Distributed Parameter" )
        ev = rcp(new ATO::TopologyFieldWeighting<EvalT,AlbanyTraits>(*p,dl));
      else
        ev = rcp(new ATO::TopologyWeighting<EvalT,AlbanyTraits>(*p,dl));

      fm0.template registerEvaluator<EvalT>(ev);

      //if(some input stuff)
      atoUtils.SaveCellStateField(fm0, stateMgr, fieldName+"_Weighted", 
                                  elementBlockName, layout, numDim);
    }
  }
  /*******************************************************************************/
  /** End topology weighting *****************************************************/
  /*******************************************************************************/

  { // Displacement Resid 
    RCP<ParameterList> p = rcp(new ParameterList("Displacement Resid"));

    p->set<bool>("Disable Transient", true);

    //Input
    if( params->isType<Teuchos::RCP<ATO::TopologyArray> > ("Topologies") )
      p->set<std::string>("Stress Name", stressName+"_Weighted");
    else 
      p->set<std::string>("Stress Name", stressName);
    p->set< RCP<DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);

    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

    //Output
    p->set<std::string>("Residual Name", "Displacement Residual");
    p->set< RCP<DataLayout> >("Node Vector Data Layout", dl->node_vector);

    ev = rcp(new LCM::ElasticityResid<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl->dummy);
    fm0.requireField<EvalT>(res_tag);

    return res_tag.clone();
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, params, stateMgr, &meshSpecs);
  }

  return Teuchos::null;
}

#endif // LINEARELASTICITYPROBLEM_HPP
