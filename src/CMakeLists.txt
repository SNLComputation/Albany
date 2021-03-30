##*****************************************************************//
##    Albany 3.0:  Copyright 2016 Sandia Corporation               //
##    This Software is released under the BSD license detailed     //
##    in the file "license.txt" in the top-level Albany directory  //
##*****************************************************************//

# Configure the file containing Albany configuration macros,
# and add its folder to the include directories
configure_file (${CMAKE_CURRENT_SOURCE_DIR}/Albany_config.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/Albany_config.h)

include(CheckCXXSourceCompiles)

set (CMAKE_REQUIRED_INCLUDES ${Trilinos_INCLUDE_DIRS})
set (CMAKE_REQUIRED_LIBRARIES ${Trilinos_LIBRARY_DIRS})
set (CMAKE_REQUIRED_FLAGS ${Trilinos_CXX_FLAGS} )

# If Albany is built against a Trilinos installation which includes pytrilinos,
# Albany should not be linked to pytrilinos. The pytrilinos dynamic library can
# undefined python related symbols.
if (";${Trilinos_LIBRARIES};" MATCHES ";pytrilinos;")
  list(REMOVE_ITEM Trilinos_LIBRARIES pytrilinos)
endif()

# Add Albany external libraries
set(ALB_TRILINOS_LIBS ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})

set (ALBANY_INCLUDE_DIRS
     ${CMAKE_CURRENT_BINARY_DIR}
     ${Albany_SOURCE_DIR}/src
     ${Albany_SOURCE_DIR}/src/evaluators
     ${Albany_SOURCE_DIR}/src/evaluators/bc
     ${Albany_SOURCE_DIR}/src/evaluators/gather
     ${Albany_SOURCE_DIR}/src/evaluators/interpolation
     ${Albany_SOURCE_DIR}/src/evaluators/pde
     ${Albany_SOURCE_DIR}/src/evaluators/response
     ${Albany_SOURCE_DIR}/src/evaluators/scatter
     ${Albany_SOURCE_DIR}/src/evaluators/state
     ${Albany_SOURCE_DIR}/src/evaluators/utility
     ${Albany_SOURCE_DIR}/src/problems
     ${Albany_SOURCE_DIR}/src/responses
     ${Albany_SOURCE_DIR}/src/disc/stk
     ${Albany_SOURCE_DIR}/src/disc
     ${Albany_SOURCE_DIR}/src/utility
)

link_directories (${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS})

set (SOURCES
  AnalyticFunction.cpp
  Albany_RegressionTests.cpp
  Albany_SolverFactory.cpp
  Albany_Utils.cpp
  PHAL_Dimension.cpp
  PHAL_Setup.cpp
  Albany_Application.cpp
  InitialCondition.cpp
  Albany_Memory.cpp
  Albany_ModelEvaluator.cpp
  Albany_NullSpaceUtils.cpp
  Albany_ObserverImpl.cpp
  Albany_PiroObserver.cpp
  Albany_StatelessObserverImpl.cpp
  Albany_StateManager.cpp
  Albany_StateInfoStruct.cpp
  PHAL_Utilities.cpp
  SolutionManager.cpp
  )

if (ALBANY_EPETRA)
  list (APPEND SOURCES
    Albany_ObserverFactory.cpp
    Petra_Converters_64.cpp
  )
endif()


if(ALBANY_EPETRA)
  list (APPEND SOURCES
    Albany_NOXObserver.cpp
    Albany_NOXStatelessObserver.cpp
  )
endif()

set(HEADERS
  AnalyticFunction.hpp
  Albany_Application.hpp
  Albany_DataTypes.hpp
  Albany_DistributedParameter.hpp
  Albany_DistributedParameterLibrary.hpp
  Albany_DistributedParameterDerivativeOp.hpp
  Albany_DummyParameterAccessor.hpp
  Albany_EigendataInfoStructT.hpp
  InitialCondition.hpp
  Albany_KokkosTypes.hpp
  Albany_Memory.hpp
  Albany_ModelEvaluator.hpp
  Albany_NullSpaceUtils.hpp
  Albany_ObserverImpl.hpp
  Albany_PiroObserver.hpp
  Albany_RegressionTests.hpp
  Albany_ScalarOrdinalTypes.hpp
  Albany_SolverFactory.hpp
  Albany_StateManager.hpp
  Albany_StateInfoStruct.hpp
  Albany_StatelessObserverImpl.hpp
  Albany_TpetraThyraTypes.hpp
  Albany_TpetraTypes.hpp
  Albany_ThyraTypes.hpp
  Albany_Utils.hpp
  PHAL_AlbanyTraits.hpp
  PHAL_Dimension.hpp
  PHAL_FactoryTraits.hpp
  PHAL_Setup.hpp
  PHAL_TypeKeyMap.hpp
  PHAL_Utilities.hpp
  PHAL_Utilities_Def.hpp
  PHAL_Workset.hpp
  SolutionManager.hpp
  )

if(ALBANY_EPETRA)
  list (APPEND HEADERS
    Albany_EigendataInfoStruct.hpp
    Albany_NOXObserver.hpp
    Albany_NOXStatelessObserver.hpp
    Albany_ObserverFactory.hpp
    Petra_Converters.hpp
  )
endif()

#utility
list (APPEND SOURCES
  utility/Counter.cpp
  utility/CounterMonitor.cpp
  utility/DisplayTable.cpp
  utility/PerformanceContext.cpp
  utility/TimeMonitor.cpp
  utility/Albany_CombineAndScatterManager.cpp
  utility/Albany_CombineAndScatterManagerTpetra.cpp
  utility/Albany_CommUtils.cpp
  utility/Albany_Gather.cpp
  utility/Albany_GlobalLocalIndexer.cpp
  utility/Albany_Hessian.cpp
  utility/Albany_ThyraCrsMatrixFactory.cpp
  utility/Albany_ThyraBlockedCrsMatrixFactory.cpp
  utility/Albany_ThyraUtils.cpp
  utility/Albany_TpetraThyraUtils.cpp
  utility/VariableMonitor.cpp
  utility/StaticAllocator.cpp
  )
list (APPEND HEADERS
  utility/Counter.hpp
  utility/CounterMonitor.hpp
  utility/DisplayTable.hpp
  utility/MonitorBase.hpp
  utility/PerformanceContext.hpp
  utility/string.hpp
  utility/TimeGuard.hpp
  utility/TimeMonitor.hpp
  utility/Albany_CombineAndScatterManager.hpp
  utility/Albany_CombineAndScatterManagerTpetra.hpp
  utility/Albany_CommUtils.hpp
  utility/Albany_Gather.hpp
  utility/Albany_GlobalLocalIndexer.hpp
  utility/Albany_GlobalLocalIndexerTpetra.hpp
  utility/Albany_Hessian.hpp
  utility/Albany_ThyraCrsMatrixFactory.hpp
  utility/Albany_ThyraBlockedCrsMatrixFactory.hpp
  utility/Albany_ThyraUtils.hpp
  utility/Albany_TpetraThyraUtils.hpp
  utility/VariableMonitor.hpp
  utility/StaticAllocator.hpp
  utility/math/Tensor.hpp
  utility/math/TensorCommon.hpp
  utility/math/TensorDetail.hpp
  utility/math/TensorImpl.hpp
  )

if (ALBANY_EPETRA)
  list (APPEND SOURCES
    utility/Albany_EpetraThyraUtils.cpp
    utility/Albany_CombineAndScatterManagerEpetra.cpp
  )
  list (APPEND HEADERS
    utility/Albany_Epetra_FECrsMatrix.hpp
    utility/Albany_EpetraThyraUtils.hpp
    utility/Albany_CombineAndScatterManagerEpetra.hpp
    utility/Albany_GlobalLocalIndexerEpetra.hpp
  )
endif()

add_executable(xml2yaml utility/xml2yaml.cpp)
add_executable(yaml2xml utility/yaml2xml.cpp)
target_link_libraries(xml2yaml teuchosparameterlist)
target_link_libraries(yaml2xml teuchosparameterlist)
target_include_directories(yaml2xml SYSTEM PUBLIC
                          "${Trilinos_INCLUDE_DIRS};${Trilinos_TPL_INCLUDE_DIRS}")
target_include_directories(xml2yaml SYSTEM PUBLIC
                          "${Trilinos_INCLUDE_DIRS};${Trilinos_TPL_INCLUDE_DIRS}")

#problems
list (APPEND SOURCES
  problems/Albany_AbstractProblem.cpp
  problems/Albany_BCUtils.cpp
  problems/Albany_EvaluatorUtils.cpp
  problems/Albany_HeatProblem.cpp
  problems/Albany_Layouts.cpp
  problems/Albany_PopulateMesh.cpp
  problems/Albany_ProblemFactory.cpp
  problems/Albany_ProblemUtils.cpp
  problems/Albany_ResponseUtilities.cpp
  problems/Albany_SideLaplacianProblem.cpp
  problems/Albany_MaterialDatabase.cpp
  )

list (APPEND HEADERS
  problems/Albany_AbstractProblem.hpp
  problems/Albany_BCUtils.hpp
  problems/Albany_BCUtils_Def.hpp
  problems/Albany_EvaluatorUtils.hpp
  problems/Albany_EvaluatorUtils_Def.hpp
  problems/Albany_GeneralPurposeFieldsNames.hpp
  problems/Albany_HeatProblem.hpp
  problems/Albany_Layouts.hpp
  problems/Albany_PopulateMesh.hpp
  problems/Albany_ProblemFactory.hpp
  problems/Albany_ProblemUtils.hpp
  problems/Albany_ResponseUtilities.hpp
  problems/Albany_ResponseUtilities_Def.hpp
  problems/Albany_SideLaplacianProblem.hpp
  problems/Albany_MaterialDatabase.hpp
  problems/Albany_ParamEnum.hpp
  )


#responses
list (APPEND SOURCES
  responses/Albany_CumulativeScalarResponseFunction.cpp
  responses/Albany_DistributedResponseFunction.cpp
  responses/Albany_FieldManagerScalarResponseFunction.cpp
  responses/Albany_FieldManagerResidualOnlyResponseFunction.cpp
  responses/Albany_SamplingBasedScalarResponseFunction.cpp
  responses/Albany_KLResponseFunction.cpp
  responses/Albany_ResponseFactory.cpp
  responses/Albany_ScalarResponseFunction.cpp
  responses/Albany_SolutionCullingStrategy.cpp
  responses/Albany_SolutionAverageResponseFunction.cpp
  responses/Albany_SolutionFileResponseFunction.cpp
  responses/Albany_SolutionMaxValueResponseFunction.cpp
  responses/Albany_SolutionMinValueResponseFunction.cpp
  responses/Albany_SolutionResponseFunction.cpp
  responses/Albany_SolutionTwoNormResponseFunction.cpp
  responses/Albany_SolutionValuesResponseFunction.cpp
  )
list (APPEND HEADERS
  responses/Albany_AbstractResponseFunction.hpp
  responses/Albany_CumulativeScalarResponseFunction.hpp
  responses/Albany_DistributedResponseFunction.hpp
  responses/Albany_FieldManagerScalarResponseFunction.hpp
  responses/Albany_KLResponseFunction.hpp
  responses/Albany_SamplingBasedScalarResponseFunction.hpp
  responses/Albany_ResponseFactory.hpp
  responses/Albany_ScalarResponseFunction.hpp
  responses/Albany_SolutionAverageResponseFunction.hpp
  responses/Albany_SolutionCullingStrategy.hpp
  responses/Albany_SolutionFileResponseFunction.hpp
  responses/Albany_SolutionMaxValueResponseFunction.hpp
  responses/Albany_SolutionMinValueResponseFunction.cpp
  responses/Albany_SolutionResponseFunction.hpp
  responses/Albany_SolutionTwoNormResponseFunction.hpp
  responses/Albany_SolutionValuesResponseFunction.hpp
  )

#discretization
list (APPEND SOURCES
  disc/Adapt_NodalDataBase.cpp
  disc/Adapt_NodalDataVector.cpp
  disc/Albany_DiscretizationFactory.cpp
  disc/Albany_MeshSpecs.cpp
  )
list (APPEND HEADERS
  disc/Adapt_NodalDataBase.hpp
  disc/Adapt_NodalDataVector.hpp
  disc/Adapt_NodalFieldUtils.hpp
  disc/Albany_DiscretizationUtils.hpp
  disc/Albany_AbstractDiscretization.hpp
  disc/Albany_AbstractFieldContainer.hpp
  disc/Albany_AbstractMeshStruct.hpp
  disc/Albany_AbstractNodeFieldContainer.hpp
  disc/Albany_DiscretizationFactory.hpp
  disc/Albany_MeshSpecs.hpp
  disc/Albany_NodalDOFManager.hpp
  )

#stk
list(APPEND SOURCES
  disc/stk/Albany_AsciiSTKMesh2D.cpp
  disc/stk/Albany_AsciiSTKMeshStruct.cpp
  disc/stk/Albany_BlockedSTKDiscretization.cpp
  disc/stk/Albany_ExtrudedSTKMeshStruct.cpp
  disc/stk/Albany_GenericSTKFieldContainer.cpp
  disc/stk/Albany_GenericSTKMeshStruct.cpp
  disc/stk/Albany_GmshSTKMeshStruct.cpp
  disc/stk/Albany_IossSTKMeshStruct.cpp
  disc/stk/Albany_MultiSTKFieldContainer.cpp
  disc/stk/Albany_OrdinarySTKFieldContainer.cpp
  disc/stk/Albany_SideSetSTKMeshStruct.cpp
  disc/stk/Albany_STKDiscretization.cpp
  disc/stk/Albany_STKFieldContainerHelper.cpp
  disc/stk/Albany_STKNodeFieldContainer.cpp
  disc/stk/Albany_STKNodeSharing.cpp
  disc/stk/Albany_STK3DPointStruct.cpp
  disc/stk/Albany_TmplSTKMeshStruct.cpp
  disc/stk/STKConnManager.cpp
)
list (APPEND HEADERS
  disc/stk/Albany_AbstractSTKFieldContainer.hpp
  disc/stk/Albany_AbstractSTKMeshStruct.hpp
  disc/stk/Albany_AsciiSTKMeshStruct.hpp
  disc/stk/Albany_AsciiSTKMesh2D.hpp
  disc/stk/Albany_BlockedSTKDiscretization.hpp
  disc/stk/Albany_ExtrudedSTKMeshStruct.hpp
  disc/stk/Albany_GenericSTKMeshStruct.hpp
  disc/stk/Albany_GmshSTKMeshStruct.hpp
  disc/stk/Albany_GenericSTKFieldContainer.hpp
  disc/stk/Albany_GenericSTKFieldContainer_Def.hpp
  disc/stk/Albany_IossSTKMeshStruct.hpp
  disc/stk/Albany_MultiSTKFieldContainer.hpp
  disc/stk/Albany_MultiSTKFieldContainer_Def.hpp
  disc/stk/Albany_NodalGraphUtils.hpp
  disc/stk/Albany_OrdinarySTKFieldContainer.hpp
  disc/stk/Albany_OrdinarySTKFieldContainer_Def.hpp
  disc/stk/Albany_SideSetSTKMeshStruct.hpp
  disc/stk/Albany_STKDiscretization.hpp
  disc/stk/Albany_STKFieldContainerHelper.hpp
  disc/stk/Albany_STKNodeFieldContainer.hpp
  disc/stk/Albany_STKNodeFieldContainer_Def.hpp
  disc/stk/Albany_STKNodeSharing.hpp
  disc/stk/Albany_STK3DPointStruct.hpp
  disc/stk/Albany_TmplSTKMeshStruct.hpp
  disc/stk/Albany_TmplSTKMeshStruct_Def.hpp
  disc/stk/STKConnManager.hpp
)

if (ALBANY_STK_REBALANCE)
  list (APPEND SOURCES
    disc/stk/percept/stk_rebalance/GeomDecomp.cpp
    disc/stk/percept/stk_rebalance/Partition.cpp
    disc/stk/percept/stk_rebalance/Rebalance.cpp
    disc/stk/percept/stk_rebalance/RebalanceUtils.cpp
    disc/stk/percept/stk_rebalance/ZoltanPartition.cpp
  )
  list (APPEND HEADERS
    disc/stk/percept/PerceptMesh.hpp
    disc/stk/percept/Percept_MyPairIterRelation.hpp
    disc/stk/percept/stk_rebalance/GeomDecomp.hpp
    disc/stk/percept/stk_rebalance/Partition.hpp
    disc/stk/percept/stk_rebalance/Rebalance.hpp
    disc/stk/percept/stk_rebalance/RebalanceUtils.hpp
    disc/stk/percept/stk_rebalance/ZoltanPartition.hpp
  )
endif()

#evaluators
list (APPEND SOURCES
  evaluators/bc/PHAL_Dirichlet.cpp
  evaluators/bc/PHAL_TimeDepDBC.cpp
  evaluators/bc/PHAL_TimeDepSDBC.cpp
  evaluators/bc/PHAL_SDirichlet.cpp
  evaluators/bc/PHAL_DirichletCoordinateFunction.cpp
  evaluators/bc/PHAL_DirichletField.cpp
  evaluators/bc/PHAL_SDirichletField.cpp
  evaluators/bc/PHAL_IdentityCoordinateFunctionTraits.cpp
  evaluators/bc/PHAL_Neumann.cpp
  evaluators/gather/PHAL_GatherAuxData.cpp
  evaluators/gather/PHAL_GatherCoordinateVector.cpp
  evaluators/gather/PHAL_GatherScalarNodalParameter.cpp
  evaluators/gather/PHAL_GatherSolution.cpp
  evaluators/gather/PHAL_GatherBlockedSolution.cpp
  evaluators/interpolation/PHAL_DOFCellToSide.cpp
  evaluators/interpolation/PHAL_DOFCellToSideQP.cpp
  evaluators/interpolation/PHAL_DOFGradInterpolation.cpp
  evaluators/interpolation/PHAL_DOFGradInterpolationSide.cpp
  evaluators/interpolation/PHAL_DOFInterpolation.cpp
  evaluators/interpolation/PHAL_DOFInterpolationSide.cpp
  evaluators/interpolation/PHAL_DOFSideToCell.cpp
  evaluators/interpolation/PHAL_DOFTensorGradInterpolation.cpp
  evaluators/interpolation/PHAL_DOFTensorInterpolation.cpp
  evaluators/interpolation/PHAL_DOFVecGradInterpolation.cpp
  evaluators/interpolation/PHAL_DOFVecGradInterpolationSide.cpp
  evaluators/interpolation/PHAL_DOFVecInterpolation.cpp
  evaluators/interpolation/PHAL_DOFVecInterpolationSide.cpp
  evaluators/interpolation/PHAL_NodesToCellInterpolation.cpp
  evaluators/interpolation/PHAL_QuadPointsToCellInterpolation.cpp
  evaluators/interpolation/PHAL_SideQuadPointsToSideInterpolation.cpp
  evaluators/pde/PHAL_HeatEqResid.cpp
  evaluators/pde/PHAL_NSMaterialProperty.cpp
  evaluators/pde/PHAL_SideLaplacianResidual.cpp
  evaluators/pde/PHAL_ThermalConductivity.cpp
  evaluators/response/PHAL_ResponseFieldIntegral.cpp
  evaluators/response/PHAL_ResponseSquaredL2Difference.cpp
  evaluators/response/PHAL_ResponseSquaredL2DifferenceSide.cpp
  evaluators/scatter/PHAL_ScatterResidual.cpp
  evaluators/scatter/PHAL_ScatterSideEqnResidual.cpp
  evaluators/scatter/PHAL_ScatterScalarNodalParameter.cpp
  evaluators/scatter/PHAL_ScatterScalarResponse.cpp
  evaluators/scatter/PHAL_SeparableScatterScalarResponse.cpp
  evaluators/state/PHAL_LoadSideSetStateField.cpp
  evaluators/state/PHAL_LoadStateField.cpp
  evaluators/state/PHAL_ReadStateField.cpp
  evaluators/state/PHAL_SaveCellStateField.cpp
  evaluators/state/PHAL_SaveNodalField.cpp
  evaluators/state/PHAL_SaveSideSetStateField.cpp
  evaluators/state/PHAL_SaveStateField.cpp
  evaluators/utility/PHAL_Absorption.cpp
  evaluators/utility/PHAL_AddNoise.cpp
  evaluators/utility/PHAL_ComputeBasisFunctions.cpp
  evaluators/utility/PHAL_ComputeBasisFunctionsSide.cpp
  evaluators/utility/PHAL_Constant.cpp
  evaluators/utility/PHAL_ConvertFieldType.cpp
  evaluators/utility/PHAL_DummyResidual.cpp
  evaluators/utility/PHAL_FieldFrobeniusNorm.cpp
  evaluators/utility/PHAL_LangevinNoiseTerm.cpp
  evaluators/utility/PHAL_MapToPhysicalFrame.cpp
  evaluators/utility/PHAL_MapToPhysicalFrameSide.cpp
  evaluators/utility/PHAL_Source.cpp
  evaluators/utility/PHAL_Field_Source.cpp
  )

if(ALBANY_STK_EXPR_EVAL)
  list (APPEND SOURCES
    evaluators/bc/PHAL_ExprEvalSDBC.cpp)
endif()

if (ALBANY_EPETRA)
  list (APPEND SOURCES
    evaluators/gather/PHAL_GatherEigenData.cpp
    evaluators/gather/PHAL_GatherEigenvectors.cpp
  )
endif()

list (APPEND HEADERS
  evaluators/bc/PHAL_Dirichlet.hpp
  evaluators/bc/PHAL_TimeDepDBC.hpp
  evaluators/bc/PHAL_TimeDepSDBC.hpp
  evaluators/bc/PHAL_SDirichlet.hpp
  evaluators/bc/PHAL_DirichletCoordinateFunction.hpp
  evaluators/bc/PHAL_DirichletCoordinateFunction_Def.hpp
  evaluators/bc/PHAL_DirichletField.hpp
  evaluators/bc/PHAL_DirichletField_Def.hpp
  evaluators/bc/PHAL_SDirichletField.hpp
  evaluators/bc/PHAL_SDirichletField_Def.hpp
  evaluators/bc/PHAL_Dirichlet_Def.hpp
  evaluators/bc/PHAL_TimeDepDBC_Def.hpp
  evaluators/bc/PHAL_TimeDepSDBC_Def.hpp
  evaluators/bc/PHAL_SDirichlet_Def.hpp
  evaluators/bc/PHAL_IdentityCoordinateFunctionTraits.hpp
  evaluators/bc/PHAL_IdentityCoordinateFunctionTraits_Def.hpp
  evaluators/bc/PHAL_Neumann.hpp
  evaluators/bc/PHAL_Neumann_Def.hpp
  evaluators/gather/PHAL_GatherAuxData.hpp
  evaluators/gather/PHAL_GatherAuxData_Def.hpp
  evaluators/gather/PHAL_GatherBlockedSolution.hpp
  evaluators/gather/PHAL_GatherBlockedSolution_Def.hpp
  evaluators/gather/PHAL_GatherCoordinateVector.hpp
  evaluators/gather/PHAL_GatherCoordinateVector_Def.hpp
  evaluators/gather/PHAL_GatherScalarNodalParameter.hpp
  evaluators/gather/PHAL_GatherScalarNodalParameter_Def.hpp
  evaluators/gather/PHAL_GatherSolution.hpp
  evaluators/gather/PHAL_GatherSolution_Def.hpp
  evaluators/interpolation/PHAL_DOFCellToSide.hpp
  evaluators/interpolation/PHAL_DOFCellToSideQP.hpp
  evaluators/interpolation/PHAL_DOFCellToSideQP_Def.hpp
  evaluators/interpolation/PHAL_DOFCellToSide_Def.hpp
  evaluators/interpolation/PHAL_DOFGradInterpolation.hpp
  evaluators/interpolation/PHAL_DOFGradInterpolationSide.hpp
  evaluators/interpolation/PHAL_DOFGradInterpolationSide_Def.hpp
  evaluators/interpolation/PHAL_DOFGradInterpolation_Def.hpp
  evaluators/interpolation/PHAL_DOFInterpolation.hpp
  evaluators/interpolation/PHAL_DOFInterpolationSide.hpp
  evaluators/interpolation/PHAL_DOFInterpolationSide_Def.hpp
  evaluators/interpolation/PHAL_DOFInterpolation_Def.hpp
  evaluators/interpolation/PHAL_DOFSideToCell.hpp
  evaluators/interpolation/PHAL_DOFSideToCell_Def.hpp
  evaluators/interpolation/PHAL_DOFTensorGradInterpolation.hpp
  evaluators/interpolation/PHAL_DOFTensorGradInterpolation_Def.hpp
  evaluators/interpolation/PHAL_DOFTensorInterpolation.hpp
  evaluators/interpolation/PHAL_DOFTensorInterpolation_Def.hpp
  evaluators/interpolation/PHAL_DOFVecGradInterpolation.hpp
  evaluators/interpolation/PHAL_DOFVecGradInterpolationSide.hpp
  evaluators/interpolation/PHAL_DOFVecGradInterpolationSide_Def.hpp
  evaluators/interpolation/PHAL_DOFVecGradInterpolation_Def.hpp
  evaluators/interpolation/PHAL_DOFVecInterpolation.hpp
  evaluators/interpolation/PHAL_DOFVecInterpolationSide.hpp
  evaluators/interpolation/PHAL_DOFVecInterpolationSide_Def.hpp
  evaluators/interpolation/PHAL_DOFVecInterpolation_Def.hpp
  evaluators/interpolation/PHAL_NodesToCellInterpolation.hpp
  evaluators/interpolation/PHAL_NodesToCellInterpolation_Def.hpp
  evaluators/interpolation/PHAL_QuadPointsToCellInterpolation.hpp
  evaluators/interpolation/PHAL_QuadPointsToCellInterpolation_Def.hpp
  evaluators/interpolation/PHAL_SideQuadPointsToSideInterpolation.hpp
  evaluators/interpolation/PHAL_SideQuadPointsToSideInterpolation_Def.hpp
  evaluators/pde/PHAL_HeatEqResid.hpp
  evaluators/pde/PHAL_HeatEqResid_Def.hpp
  evaluators/pde/PHAL_NSMaterialProperty.hpp
  evaluators/pde/PHAL_NSMaterialProperty_Def.hpp
  evaluators/pde/PHAL_SideLaplacianResidual.hpp
  evaluators/pde/PHAL_SideLaplacianResidual_Def.hpp
  evaluators/pde/PHAL_ThermalConductivity.hpp
  evaluators/pde/PHAL_ThermalConductivity_Def.hpp
  evaluators/utility/PHAL_SharedParameter.hpp
  evaluators/response/PHAL_ResponseFieldIntegral.hpp
  evaluators/response/PHAL_ResponseFieldIntegral_Def.hpp
  evaluators/response/PHAL_ResponseSquaredL2Difference.hpp
  evaluators/response/PHAL_ResponseSquaredL2DifferenceSide.hpp
  evaluators/response/PHAL_ResponseSquaredL2DifferenceSide_Def.hpp
  evaluators/response/PHAL_ResponseSquaredL2Difference_Def.hpp
  evaluators/scatter/PHAL_ScatterResidual.hpp
  evaluators/scatter/PHAL_ScatterResidual_Def.hpp
  evaluators/scatter/PHAL_ScatterSideEqnResidual.hpp
  evaluators/scatter/PHAL_ScatterSideEqnResidual_Def.hpp
  evaluators/scatter/PHAL_ScatterScalarNodalParameter.hpp
  evaluators/scatter/PHAL_ScatterScalarNodalParameter_Def.hpp
  evaluators/scatter/PHAL_ScatterScalarResponse.hpp
  evaluators/scatter/PHAL_ScatterScalarResponse_Def.hpp
  evaluators/state/PHAL_LoadSideSetStateField.hpp
  evaluators/state/PHAL_LoadSideSetStateField_Def.hpp
  evaluators/state/PHAL_LoadStateField.hpp
  evaluators/state/PHAL_LoadStateField_Def.hpp
  evaluators/state/PHAL_ReadStateField.hpp
  evaluators/state/PHAL_ReadStateField_Def.hpp
  evaluators/state/PHAL_SaveCellStateField.hpp
  evaluators/state/PHAL_SaveCellStateField_Def.hpp
  evaluators/state/PHAL_SaveNodalField.hpp
  evaluators/state/PHAL_SaveNodalField_Def.hpp
  evaluators/state/PHAL_SaveSideSetStateField.hpp
  evaluators/state/PHAL_SaveSideSetStateField_Def.hpp
  evaluators/state/PHAL_SaveStateField.hpp
  evaluators/state/PHAL_SaveStateField_Def.hpp
  evaluators/utility/PHAL_Absorption.hpp
  evaluators/utility/PHAL_Absorption_Def.hpp
  evaluators/utility/PHAL_AddNoise.hpp
  evaluators/utility/PHAL_AddNoise_Def.hpp
  evaluators/utility/PHAL_ComputeBasisFunctions.hpp
  evaluators/utility/PHAL_ComputeBasisFunctions_Def.hpp
  evaluators/utility/PHAL_ComputeBasisFunctionsSide.hpp
  evaluators/utility/PHAL_ComputeBasisFunctionsSide_Def.hpp
  evaluators/utility/PHAL_Constant.hpp
  evaluators/utility/PHAL_Constant_Def.hpp
  evaluators/utility/PHAL_ConvertFieldType.hpp
  evaluators/utility/PHAL_ConvertFieldType_Def.hpp
  evaluators/utility/PHAL_DummyResidual.hpp
  evaluators/utility/PHAL_DummyResidual_Def.hpp
  evaluators/utility/PHAL_FieldFrobeniusNorm.hpp
  evaluators/utility/PHAL_FieldFrobeniusNorm_Def.hpp
  evaluators/utility/PHAL_LangevinNoiseTerm.hpp
  evaluators/utility/PHAL_LangevinNoiseTerm_Def.hpp
  evaluators/utility/PHAL_MapToPhysicalFrame.hpp
  evaluators/utility/PHAL_MapToPhysicalFrameSide.hpp
  evaluators/utility/PHAL_MapToPhysicalFrameSide_Def.hpp
  evaluators/utility/PHAL_MapToPhysicalFrame_Def.hpp
  evaluators/utility/PHAL_Source.hpp
  evaluators/utility/PHAL_Source_Def.hpp
  evaluators/utility/PHAL_Field_Source.hpp
  evaluators/utility/PHAL_Field_Source_Def.hpp
  )

if(ALBANY_STK_EXPR_EVAL)
  list (APPEND HEADERS
    evaluators/bc/PHAL_ExprEvalSDBC.hpp
    evaluators/bc/PHAL_ExprEvalSDBC_Def.hpp)
endif()

if (ALBANY_EPETRA)
  list (APPEND HEADERS
    evaluators/gather/PHAL_GatherEigenvectors.hpp
    evaluators/gather/PHAL_GatherEigenvectors_Def.hpp
    evaluators/gather/PHAL_GatherEigenData.hpp
    evaluators/gather/PHAL_GatherEigenData_Def.hpp
    evaluators/scatter/PHAL_SeparableScatterScalarResponse.hpp
    evaluators/scatter/PHAL_SeparableScatterScalarResponse_Def.hpp
  )
endif()

add_library(albanyLib ${Albany_LIBRARY_TYPE} ${SOURCES})
set_target_properties(albanyLib PROPERTIES PUBLIC_HEADER "${HEADERS}")
target_link_libraries(albanyLib ${Trilinos_LIBRARIES})
if (ALBANY_SUPPRESS_TRILINOS_WARNINGS)
  target_include_directories(albanyLib SYSTEM PUBLIC
                            "${Trilinos_INCLUDE_DIRS};${Trilinos_TPL_INCLUDE_DIRS}")
else()
  target_include_directories(albanyLib PUBLIC
                            "${Trilinos_INCLUDE_DIRS};${Trilinos_TPL_INCLUDE_DIRS}")
endif()
target_include_directories(albanyLib PUBLIC
                            $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>
                           "$<BUILD_INTERFACE:${ALBANY_INCLUDE_DIRS}>")

# Note: Albany_EXTRA_LIBRARIES is to allow users to specify
# libraries that one cannot get out of Trilinos directly.
set(TPL_LIBRARIES
  ${ALB_TRILINOS_LIBS}
  ${Trilinos_EXTRA_LD_FLAGS}
  ${Albany_EXTRA_LIBRARIES}
  ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}
  )

# Add Albany physics libraries, as enabled.

set(ALBANY_LIBRARIES albanyLib)

if(ALBANY_DEMO_PDES)
  set(PDE_SRCS
    problems/Albany_AdvDiffProblem.cpp
    problems/Albany_ReactDiffSystem.cpp
    problems/Albany_CahnHillProblem.cpp
    problems/Albany_ComprNSProblem.cpp
    problems/Albany_DemoProblemFactory.cpp
    problems/Albany_Helmholtz2DProblem.cpp
    problems/Albany_LinComprNSProblem.cpp
    problems/Albany_ODEProblem.cpp
    problems/Albany_NavierStokes.cpp
    problems/Albany_PNPProblem.cpp
    problems/Albany_ThermoElectrostaticsProblem.cpp
    problems/Albany_ThermalProblem.cpp
    evaluators/pde/PHAL_AdvDiffResid.cpp
    evaluators/pde/PHAL_ReactDiffSystemResid.cpp
    evaluators/pde/PHAL_CahnHillChemTerm.cpp
    evaluators/pde/PHAL_CahnHillRhoResid.cpp
    evaluators/pde/PHAL_CahnHillWResid.cpp
    evaluators/pde/PHAL_ComprNSBodyForce.cpp
    evaluators/pde/PHAL_ComprNSResid.cpp
    evaluators/pde/PHAL_ComprNSViscosity.cpp
    evaluators/pde/PHAL_HelmholtzResid.cpp
    evaluators/pde/PHAL_JouleHeating.cpp
    evaluators/pde/PHAL_LinComprNSBodyForce.cpp
    evaluators/pde/PHAL_LinComprNSResid.cpp
    evaluators/pde/PHAL_NSBodyForce.cpp
    evaluators/pde/PHAL_NSContinuityResid.cpp
    evaluators/pde/PHAL_NSContravarientMetricTensor.cpp
    evaluators/pde/PHAL_NSForchheimerTerm.cpp
    evaluators/pde/PHAL_NSMomentumResid.cpp
    evaluators/pde/PHAL_NSNeutronEqResid.cpp
    evaluators/pde/PHAL_NSPermeabilityTerm.cpp
    evaluators/pde/PHAL_ThermalResid.cpp
    evaluators/pde/PHAL_NSRm.cpp
    evaluators/pde/PHAL_NSTauM.cpp
    evaluators/pde/PHAL_NSTauT.cpp
    evaluators/pde/PHAL_NSThermalEqResid.cpp
    evaluators/pde/PHAL_ODEResid.cpp
    evaluators/pde/PHAL_PoissonResid.cpp
    evaluators/pde/PHAL_Permittivity.cpp
    evaluators/pde/PHAL_TEProp.cpp
    evaluators/pde/PNP_ConcentrationResid.cpp
    evaluators/pde/PNP_PotentialResid.cpp
  )

  set(PDE_HDRS
    problems/Albany_AdvDiffProblem.hpp
    problems/Albany_ReactDiffSystem.hpp
    problems/Albany_CahnHillProblem.hpp
    problems/Albany_ComprNSProblem.hpp
    problems/Albany_Helmholtz2DProblem.hpp
    problems/Albany_LinComprNSProblem.hpp
    problems/Albany_NavierStokes.hpp
    problems/Albany_ODEProblem.hpp
    problems/Albany_PNPProblem.hpp
    problems/Albany_ThermoElectrostaticsProblem.hpp
    problems/Albany_ThermalProblem.hpp
    evaluators/pde/PHAL_AdvDiffResid.hpp
    evaluators/pde/PHAL_AdvDiffResid_Def.hpp
    evaluators/pde/PHAL_ReactDiffSystemResid.hpp
    evaluators/pde/PHAL_ReactDiffSystemResid_Def.hpp
    evaluators/pde/PHAL_CahnHillRhoResid.hpp
    evaluators/pde/PHAL_CahnHillRhoResid_Def.hpp
    evaluators/pde/PHAL_CahnHillWResid.hpp
    evaluators/pde/PHAL_CahnHillWResid_Def.hpp
    evaluators/pde/PHAL_CahnHillChemTerm.hpp
    evaluators/pde/PHAL_CahnHillChemTerm_Def.hpp
    evaluators/pde/PHAL_ComprNSBodyForce.hpp
    evaluators/pde/PHAL_ComprNSBodyForce_Def.hpp
    evaluators/pde/PHAL_ComprNSResid.hpp
    evaluators/pde/PHAL_ComprNSResid_Def.hpp
    evaluators/pde/PHAL_ComprNSViscosity.hpp
    evaluators/pde/PHAL_ComprNSViscosity_Def.hpp
    evaluators/pde/PHAL_HelmholtzResid.hpp
    evaluators/pde/PHAL_HelmholtzResid_Def.hpp
    evaluators/pde/PHAL_JouleHeating.hpp
    evaluators/pde/PHAL_JouleHeating_Def.hpp
    evaluators/pde/PHAL_LinComprNSBodyForce.hpp
    evaluators/pde/PHAL_LinComprNSBodyForce_Def.hpp
    evaluators/pde/PHAL_LinComprNSResid.hpp
    evaluators/pde/PHAL_LinComprNSResid_Def.hpp
    evaluators/pde/PHAL_NSContinuityResid.hpp
    evaluators/pde/PHAL_NSContinuityResid_Def.hpp
    evaluators/pde/PHAL_NSBodyForce.hpp
    evaluators/pde/PHAL_NSBodyForce_Def.hpp
    evaluators/pde/PHAL_NSContravarientMetricTensor.hpp
    evaluators/pde/PHAL_NSContravarientMetricTensor_Def.hpp
    evaluators/pde/PHAL_NSForchheimerTerm.hpp
    evaluators/pde/PHAL_NSForchheimerTerm_Def.hpp
    evaluators/pde/PHAL_NSMomentumResid.hpp
    evaluators/pde/PHAL_NSMomentumResid_Def.hpp
    evaluators/pde/PHAL_NSNeutronEqResid.hpp
    evaluators/pde/PHAL_NSNeutronEqResid_Def.hpp
    evaluators/pde/PHAL_NSPermeabilityTerm.hpp
    evaluators/pde/PHAL_NSPermeabilityTerm_Def.hpp
    evaluators/pde/PHAL_NSRm.hpp
    evaluators/pde/PHAL_NSRm_Def.hpp
    evaluators/pde/PHAL_NSTauM.hpp
    evaluators/pde/PHAL_NSTauM_Def.hpp
    evaluators/pde/PHAL_NSTauT.hpp
    evaluators/pde/PHAL_NSTauT_Def.hpp
    evaluators/pde/PHAL_NSThermalEqResid.hpp
    evaluators/pde/PHAL_NSThermalEqResid_Def.hpp
    evaluators/pde/PHAL_ODEResid.hpp
    evaluators/pde/PHAL_ODEResid_Def.hpp
    evaluators/pde/PHAL_PoissonResid.hpp
    evaluators/pde/PHAL_PoissonResid_Def.hpp
    evaluators/pde/PHAL_Permittivity.hpp
    evaluators/pde/PHAL_Permittivity_Def.hpp
    evaluators/pde/PHAL_TEProp.hpp
    evaluators/pde/PHAL_TEProp_Def.hpp
    evaluators/pde/PNP_ConcentrationResid.hpp
    evaluators/pde/PNP_ConcentrationResid_Def.hpp
    evaluators/pde/PNP_PotentialResid.hpp
    evaluators/pde/PNP_PotentialResid_Def.hpp
    evaluators/pde/PHAL_ThermalResid.hpp
    evaluators/pde/PHAL_ThermalResid_Def.hpp
  )

  add_library(albanyPDEs ${Albany_LIBRARY_TYPE} ${PDE_SRCS})
  set_target_properties(albanyPDEs PROPERTIES PUBLIC_HEADER "${PDE_HDRS}")
  target_link_libraries(albanyPDEs PUBLIC albanyLib)
  if (ALBANY_SUPPRESS_TRILINOS_WARNINGS)
    target_include_directories(albanyPDEs SYSTEM PUBLIC
                              "${Trilinos_INCLUDE_DIRS};${Trilinos_TPL_INCLUDE_DIRS}")
  else()
    target_include_directories(albanyPDEs PUBLIC
                              "${Trilinos_INCLUDE_DIRS};${Trilinos_TPL_INCLUDE_DIRS}")
  endif()
  target_include_directories(albanyPDEs PUBLIC
                             "$<BUILD_INTERFACE:${ALBANY_INCLUDE_DIRS}>"
                              $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>)
  list(APPEND ALBANY_LIBRARIES albanyPDEs)
endif()

if (ALBANY_LANDICE)
  add_subdirectory(LandIce)
  list (APPEND ALBANY_LIBRARIES LandIce)

  if (ENABLE_MPAS_INTERFACE)
    list (APPEND ALBANY_LIBRARIES mpasInterface)
  endif()

  if (ENABLE_CISM_INTERFACE)
    list (APPEND ALBANY_LIBRARIES cismInterface)
  endif()
endif()

# Now build executables from this library, tailored main()s, and Trilinos
if (NOT ALBANY_LIBRARIES_ONLY)

  message("\n-- Albany Executables: Albany; AlbanyAnalysis")

  add_executable(Albany Main_Solve.cpp)
  set(ALBANY_EXECUTABLES Albany)

  add_executable(AlbanyAnalysis Main_Analysis.cpp)
  set(ALBANY_EXECUTABLES ${ALBANY_EXECUTABLES} AlbanyAnalysis)

endif (NOT ALBANY_LIBRARIES_ONLY)

foreach(ALB_EXEC ${ALBANY_EXECUTABLES})

  if (Albany_BUILD_STATIC_EXE)
    set_target_properties(${ALB_EXEC} PROPERTIES LINK_SEARCH_START_STATIC 1)
  endif()
  target_link_libraries(${ALB_EXEC} ${ALBANY_LIBRARIES} ${TPL_LIBRARIES})
  target_include_directories(${ALB_EXEC} PUBLIC
      $<BUILD_INTERFACE:${ALBANY_INCLUDE_DIRS}>
      $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>)
  # We may not want warnings from Trilinos and/or its TPLs...
  if (ALBANY_SUPPRESS_TRILINOS_WARNINGS)
    target_include_directories(${ALB_EXEC} SYSTEM PUBLIC
                              "${Trilinos_INCLUDE_DIRS};${Trilinos_TPL_INCLUDE_DIRS}")
  else()
    target_include_directories(${ALB_EXEC} PUBLIC
                              "${Trilinos_INCLUDE_DIRS};${Trilinos_TPL_INCLUDE_DIRS}")
  endif()
endforeach()

if (INSTALL_ALBANY)
  configure_package_config_file(AlbanyConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/AlbanyConfig.cmake
    INSTALL_DESTINATION ${LIB_INSTALL_DIR}/Albany/cmake
    PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR)
  install(FILES ${HEADERS}
    DESTINATION ${INCLUDE_INSTALL_DIR})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Albany_config.h
    DESTINATION ${INCLUDE_INSTALL_DIR})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/AlbanyConfig.cmake
    DESTINATION ${LIB_INSTALL_DIR}/Albany/cmake )
  # Install each binary in the list
  install(TARGETS ${ALBANY_EXECUTABLES} RUNTIME DESTINATION
     ${BINARY_INSTALL_DIR})
  install(TARGETS albanyLib EXPORT albany-export
    LIBRARY DESTINATION ${LIB_INSTALL_DIR}
    ARCHIVE DESTINATION ${LIB_INSTALL_DIR})
  if (ENABLE_DEMO_PDES)
    install(TARGETS albanyPDEs EXPORT albany-export
      LIBRARY DESTINATION ${LIB_INSTALL_DIR}
      ARCHIVE DESTINATION ${LIB_INSTALL_DIR})
  endif()
  install(EXPORT albany-export DESTINATION ${LIB_INSTALL_DIR}/Albany/cmake  FILE "albany-targets.cmake")

  # Reverse albany libraries, so that they link correctly.
  # Recall, -lA -lB means B *cannot* need stuff from A.
  list (REVERSE ALBANY_LIBRARIES)

  set(TMP1 ${ALBANY_LIBRARIES} ${TPL_LIBRARIES})

  message("-- Exporting link libs to: ${CMAKE_INSTALL_PREFIX}/export_albany.in")
  string(REGEX REPLACE ";/" " /" TMP2 "${TMP1}")
  string(REPLACE ";" " -l" ALBANY_LINK_LIBS
        "-L${CMAKE_INSTALL_PREFIX}/${LIB_INSTALL_DIR}"
        " -L${Trilinos_LIBRARY_DIRS}" " -l${TMP2}")
  configure_file(export_albany.in ${CMAKE_INSTALL_PREFIX}/export_albany.in)

endif()
