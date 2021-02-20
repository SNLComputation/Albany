//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "PHAL_SaveSideSetStateField.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_AbstractSTKMeshStruct.hpp"
#include "Albany_SideSetSTKMeshStruct.hpp"
#include "Albany_AbstractSTKFieldContainer.hpp"

namespace PHAL
{

template<typename EvalT, typename Traits>
SaveSideSetStateField<EvalT, Traits>::
SaveSideSetStateField (const Teuchos::ParameterList& /* p */,
                       const Teuchos::RCP<Albany::Layouts>& /* dl */)
{
  // States Not Saved for Generic Type, only Specializations
  this->setName("Save Side Set State Field"+PHX::print<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits>
void SaveSideSetStateField<EvalT, Traits>::
postRegistrationSetup (typename Traits::SetupData /* d */,
                       PHX::FieldManager<Traits>& /* fm */)
{
  // States Not Saved for Generic Type, only Specializations
}

// **********************************************************************
template<typename EvalT, typename Traits>
void SaveSideSetStateField<EvalT, Traits>::evaluateFields(typename Traits::EvalData /* workset */)
{
  // States Not Saved for Generic Type, only Specializations
}
// **********************************************************************

// **********************************************************************
template<typename Traits>
SaveSideSetStateField<PHAL::AlbanyTraits::Residual, Traits>::
SaveSideSetStateField (const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl)
{
  sideSetName   = p.get<std::string>("Side Set Name");

  fieldName = p.get<std::string>("Field Name");
  stateName = p.get<std::string>("State Name");

  field = decltype(field)(fieldName, p.get<Teuchos::RCP<PHX::DataLayout> >("Field Layout") );

  nodalState = p.isParameter("Nodal State") ? p.get<bool>("Nodal State") : false;

  savestate_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(fieldName, dl->dummy));

  useCollapsedSidesets = dl->useCollapsedSidesets;

  this->addDependentField (field.fieldTag());
  this->addEvaluatedField (*savestate_operation);

  if (nodalState)
  {
    if (useCollapsedSidesets) {
      TEUCHOS_TEST_FOR_EXCEPTION(field.fieldTag().dataLayout().size()<2, Teuchos::Exceptions::InvalidParameter,
                                  "Error! To save a side-set nodal state, pass the cell-side-based version of it (<Cell,Side,Node,...>).\n");
      TEUCHOS_TEST_FOR_EXCEPTION(field.fieldTag().dataLayout().name(1)!="Node", Teuchos::Exceptions::InvalidParameter,
                                  "Error! To save a side-set nodal state, the third tag of the layout MUST be 'Node'.\n");
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(field.fieldTag().dataLayout().size()<3, Teuchos::Exceptions::InvalidParameter,
                                  "Error! To save a side-set nodal state, pass the cell-side-based version of it (<Cell,Side,Node,...>).\n");
      TEUCHOS_TEST_FOR_EXCEPTION(field.fieldTag().dataLayout().name(2)!="Node", Teuchos::Exceptions::InvalidParameter,
                                  "Error! To save a side-set nodal state, the third tag of the layout MUST be 'Node'.\n");
    }

    Teuchos::RCP<shards::CellTopology> cellType;
    cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");

    int sideDim = cellType->getDimension()-1;
    int numSides = cellType->getSideCount();
    int nodeMax = 0;
    for (int side=0; side<numSides; ++side) {
      int thisSideNodes = cellType->getNodeCount(sideDim,side);
      nodeMax = std::max(nodeMax, thisSideNodes);
    }
    sideNodes = Kokkos::View<int**, PHX::Device>("sideNodes", numSides, nodeMax);
    for (int side=0; side<numSides; ++side) {
      // Need to get the subcell exact count, since different sides may have different number of nodes (e.g., Wedge)
      int thisSideNodes = cellType->getNodeCount(sideDim,side);
      for (int node=0; node<thisSideNodes; ++node) {
        sideNodes(side,node) = cellType->getNodeMap(sideDim,side,node);
      }
    }
  }

  this->setName ("Save Side Set Field " + fieldName + " to Side Set State " + stateName + " <Residual>");
}

// **********************************************************************
template<typename Traits>
void SaveSideSetStateField<PHAL::AlbanyTraits::Residual, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field,fm);
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}
// **********************************************************************
template<typename Traits>
void SaveSideSetStateField<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  if (this->nodalState)
    saveNodeState(workset);
  else
    saveElemState(workset);
}

template<typename Traits>
void SaveSideSetStateField<PHAL::AlbanyTraits::Residual, Traits>::
saveElemState(typename Traits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (workset.sideSets==Teuchos::null, std::logic_error,
                              "Error! The mesh does not store any side set.\n");

  if (workset.sideSetViews->find(sideSetName)==workset.sideSetViews->end()) return; // Side set not present in this workset

  TEUCHOS_TEST_FOR_EXCEPTION (workset.disc==Teuchos::null, std::logic_error,
                              "Error! The workset must store a valid discretization pointer.\n");

  const Albany::AbstractDiscretization::SideSetDiscretizationsType& ssDiscs = workset.disc->getSideSetDiscretizations();

  TEUCHOS_TEST_FOR_EXCEPTION (ssDiscs.size()==0, std::logic_error,
                              "Error! The discretization must store side set discretizations.\n");

  TEUCHOS_TEST_FOR_EXCEPTION (ssDiscs.find(sideSetName)==ssDiscs.end(), std::logic_error,
                              "Error! No discretization found for side set " << sideSetName << ".\n");

  Teuchos::RCP<Albany::AbstractDiscretization> ss_disc = ssDiscs.at(sideSetName);

  TEUCHOS_TEST_FOR_EXCEPTION (ss_disc==Teuchos::null, std::logic_error,
                              "Error! Side discretization is invalid for side set " << sideSetName << ".\n");

  const std::map<std::string,std::map<GO,GO> >& ss_maps = workset.disc->getSideToSideSetCellMap();

  TEUCHOS_TEST_FOR_EXCEPTION (ss_maps.find(sideSetName)==ss_maps.end(), std::logic_error,
                              "Error! Something is off: the mesh has side discretization but no sideId-to-sideSetElemId map.\n");

  const std::map<GO,GO>& ss_map = ss_maps.at(sideSetName);

  // Get states from STK mesh
  Albany::StateArrays& state_arrays = ss_disc->getStateArrays();
  Albany::StateArrayVec& esa = state_arrays.elemStateArrays;
  Albany::WsLIDList& elemGIDws3D = workset.disc->getElemGIDws();
  Albany::WsLIDList& elemGIDws2D = ss_disc->getElemGIDws();

  // Get side_node->side_set_cell_node map from discretization
  TEUCHOS_TEST_FOR_EXCEPTION (workset.disc->getSideNodeNumerationMap().find(sideSetName)==workset.disc->getSideNodeNumerationMap().end(),
                              std::logic_error, "Error! Sideset " << sideSetName << " has no sideNodeNumeration map.\n");
  const std::map<GO,std::vector<int>>& sideNodeNumerationMap = workset.disc->getSideNodeNumerationMap().at(sideSetName);

  // Establishing the kind of field layout
  std::vector<PHX::DataLayout::size_type> dims;
  field.dimensions(dims);
  const std::string& tag1 = dims.size()>1 ? field.fieldTag().dataLayout().name(1) : "";
  const std::string& tag2 = dims.size()>2 ? field.fieldTag().dataLayout().name(2) : "";
  TEUCHOS_TEST_FOR_EXCEPTION (useCollapsedSidesets && dims.size()>1 && tag1!="Node" && tag1!="Dim" && tag1!="VecDim", std::logic_error,
                              "Error! Invalid field layout in SaveSideSetStateField.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (!useCollapsedSidesets && dims.size()>2 && tag2!="Node" && tag2!="Dim" && tag2!="VecDim", std::logic_error,
                              "Error! Invalid field layout in SaveSideSetStateField.\n");

  // Loop on the sides of this sideSet that are in this workset
  sideSet = workset.sideSetViews->at(sideSetName);
  for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
  {
    // Get the data that corresponds to the side
    const int elem_GID = sideSet.elem_GID(sideSet_idx);
    const int side_GID = sideSet.side_GID(sideSet_idx);
    const int cell = sideSet.elem_LID(sideSet_idx);
    const int side = sideSet.side_local_id(sideSet_idx);

    // Not sure if this is even possible, but just for debug pourposes
    TEUCHOS_TEST_FOR_EXCEPTION (elemGIDws3D[ elem_GID ].ws != (int) workset.wsIndex, std::logic_error,
                                "Error! This workset has a side that belongs to an element not in the workset.\n");

    // We know the side ID, so we can fetch two things:
    //    1) the 2D-wsIndex where the 2D element lies
    //    2) the LID of the 2D element

    TEUCHOS_TEST_FOR_EXCEPTION (ss_map.find(side_GID)==ss_map.end(), std::logic_error,
                                "Error! The sideId-to-sideSetElemId map does not store this side GID. Weird, should never happen.\n");

    int ss_cell_GID = ss_map.at(side_GID);
    int wsIndex2D = elemGIDws2D[ss_cell_GID].ws;
    int ss_cell = elemGIDws2D[ss_cell_GID].LID;

    // Then, after a safety check, we extract the StateArray of the desired state in the right 2D-ws
    TEUCHOS_TEST_FOR_EXCEPTION (esa[wsIndex2D].find(stateName) == esa[wsIndex2D].end(), std::logic_error,
                                "Error! Cannot locate " << stateName << " in PHAL_SaveSideSetStateField_Def.\n");
    Albany::MDArray state = esa[wsIndex2D].at(stateName);

    const std::vector<int>& nodeMap = sideNodeNumerationMap.at(side_GID);

    // Now we have the two arrays: 3D and 2D. We need to take the part we need from the 3D
    // and put it in the 2D one

    field.dimensions(dims);
    int size = dims.size();

    if (useCollapsedSidesets) {
      switch (size)
      {
        case 1:
          // side set cell scalar
          state(ss_cell) = field(sideSet_idx);
          break;

        case 2:
          if (tag1=="Node")
          {
            // side set node scalar
            for (unsigned int node=0; node<dims[1]; ++node)
            {
              state(ss_cell,nodeMap[node]) = field(sideSet_idx,node);
            }
          } else {
            // side set cell vector/gradient
            for (unsigned int idim=0; idim<dims[1]; ++idim)
            {
              state(ss_cell,(int) idim) = field(sideSet_idx,idim);
            }
          }
          break;

        case 3:
          if (tag1=="Node")
          {
            // side set node vector/gradient
            for (unsigned int node=0; node<dims[1]; ++node)
            {
              for (unsigned int dim=0; dim<dims[2]; ++dim)
                state(ss_cell,nodeMap[node],(int) dim) = field(sideSet_idx,node,dim);
            }
          }
          else
          {
            // side set cell tensor
            for (unsigned int idim=0; idim<dims[1]; ++idim)
            {
              for (unsigned int jdim=0; jdim<dims[2]; ++jdim)
                state(ss_cell,(int) idim,(int) jdim) = field(sideSet_idx,idim,jdim);
            }
          }
          break;

        default:
          TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
                                      "Error! Unexpected array dimensions in SaveSideSetStateField: " << size << ".\n");
      }
    } else {
      switch (size)
      {
        case 2:
          // side set cell scalar
          state(ss_cell) = field(cell,side);
          break;

        case 3:
          if (tag2=="Node")
          {
            // side set node scalar
            for (unsigned int node=0; node<dims[2]; ++node)
            {
              state(ss_cell,nodeMap[node]) = field(cell,side,node);
            }
          } else {
            // side set cell vector/gradient
            for (unsigned int idim=0; idim<dims[2]; ++idim)
            {
              state(ss_cell,(int) idim) = field(cell,side,idim);
            }
          }
          break;

        case 4:
          if (tag2=="Node")
          {
            // side set node vector/gradient
            for (unsigned int node=0; node<dims[2]; ++node)
            {
              for (unsigned int dim=0; dim<dims[3]; ++dim)
                state(ss_cell,nodeMap[node],(int) dim) = field(cell,side,node,dim);
            }
          }
          else
          {
            // side set cell tensor
            for (unsigned int idim=0; idim<dims[2]; ++idim)
            {
              for (unsigned int jdim=0; jdim<dims[3]; ++jdim)
                state(ss_cell,(int) idim,(int) jdim) = field(cell,side,idim,jdim);
            }
          }
          break;

        default:
          TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error,
                                      "Error! Unexpected array dimensions in SaveSideSetStateField: " << size << ".\n");
      }
    }

  }
}

template<typename Traits>
void SaveSideSetStateField<PHAL::AlbanyTraits::Residual, Traits>::
saveNodeState(typename Traits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (workset.sideSets==Teuchos::null, std::logic_error,
                              "Error! The mesh does not store any side set.\n");

  if (workset.sideSetViews->find(sideSetName)==workset.sideSetViews->end()) return; // Side set not present in this workset

  // Note: to save nodal fields, we need to open up the mesh, and work directly on it.
  //       For this reason, we assume the mesh is stk. Moreover, since the cell buckets
  //       and node buckets do not coincide (elem-bucket 12 does not necessarily contain
  //       all nodes in node-bucket 12), we need to work with stk fields. To do this we
  //       must extract entities from the bulk data and use them to access the values
  //       of the stk field.

  TEUCHOS_TEST_FOR_EXCEPTION (workset.disc==Teuchos::null, std::logic_error,
                              "Error! The workset must store a valid discretization pointer.\n");
  Teuchos::RCP<Albany::AbstractDiscretization> disc = workset.disc;

  Teuchos::RCP<Albany::LayeredMeshNumbering<GO>> layeredMeshNumbering = disc->getLayeredMeshNumbering();

  const Albany::AbstractDiscretization::SideSetDiscretizationsType& ssDiscs = disc->getSideSetDiscretizations();

  TEUCHOS_TEST_FOR_EXCEPTION (ssDiscs.size()==0, std::logic_error,
                              "Error! The discretization must store side set discretizations.\n");

  TEUCHOS_TEST_FOR_EXCEPTION (ssDiscs.find(sideSetName)==ssDiscs.end(), std::logic_error,
                              "Error! No discretization found for side set " << sideSetName << ".\n");

  Teuchos::RCP<Albany::AbstractDiscretization> ss_disc = ssDiscs.at(sideSetName);

  const std::map<std::string,std::map<GO,GO> >& ss_maps = disc->getSideToSideSetCellMap();

  TEUCHOS_TEST_FOR_EXCEPTION (ss_maps.find(sideSetName)==ss_maps.end(), std::logic_error,
                              "Error! Something is off: the mesh has side discretization but no sideId-to-sideSetElemId map.\n");

  // Get side_node->side_set_cell_node map from discretization
  TEUCHOS_TEST_FOR_EXCEPTION (disc->getSideNodeNumerationMap().find(sideSetName)==disc->getSideNodeNumerationMap().end(),
                              std::logic_error, "Error! Sideset " << sideSetName << " has no sideNodeNumeration map.\n");

  Teuchos::RCP<Albany::AbstractSTKMeshStruct> mesh = Teuchos::rcp_dynamic_cast<Albany::AbstractSTKMeshStruct>(ss_disc->getMeshStruct());
  TEUCHOS_TEST_FOR_EXCEPTION (mesh==Teuchos::null, std::runtime_error, "Error! Save nodal states available only for stk meshes.\n");

  stk::mesh::MetaData& metaData = *mesh->metaData;
  stk::mesh::BulkData& bulkData = *mesh->bulkData;

  const auto& ElNodeID = disc->getWsElNodeID()[workset.wsIndex];

  typedef Albany::AbstractSTKFieldContainer::ScalarFieldType SFT;
  typedef Albany::AbstractSTKFieldContainer::VectorFieldType VFT;
  SFT* scalar_field;
  VFT* vector_field;

  std::vector<PHX::DataLayout::size_type> dims;
  field.dimensions(dims);

  GO nodeId3d;
  double* values;
  stk::mesh::Entity e;

  // Notice: in the following, we retrieve the id of the stk node using the 3d mesh, since it's easier.
  //         However, the node id in the 3d mesh is guaranteed to coincide with the node id in the 2d
  //         mesh for SideSetSTKMeshStruct. If the ss mesh type is not SideSetSTKMeshStruct, then the
  //         side mesh was not extracted from the 3d one; it's either the basal mesh of an extruded mesh,
  //         or the basal and volume meshes were loaded separately. Either way, we check if the mesh is
  //         layered (it will be for extruded meshes): if not, 2d and 3d ids will coincide; if yes, they
  //         will coincide if the ordering is LAYER, and won't coincide if the ordering is COLUMN.
  //         In the latter case, we need to use the LayeredMeshNumbering structure to determine the id
  //         of the 2d node given that of the 3d node.

  if (Teuchos::rcp_dynamic_cast<Albany::SideSetSTKMeshStruct>(mesh)!=Teuchos::null ||
      layeredMeshNumbering==Teuchos::null || layeredMeshNumbering->ordering==Albany::LayeredMeshOrdering::LAYER)
  {
    // Either not a layered mesh or layered but not column-wise. Either way, the GID of the side-set's
    // nodes will coincide with the GID of the 3D mesh nodes.

    // Loop on the sides of this sideSet that are in this workset
    sideSet = workset.sideSetViews->at(sideSetName);
    if (useCollapsedSidesets) {
      for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
      {
        // Get the data that corresponds to the side
        const int cell = sideSet.elem_LID(sideSet_idx);
        const int side = sideSet.side_local_id(sideSet_idx);

        // Notice: in the following, we retrieve the id of the stk node using the 3d mesh.
        //         This is because the id of entities is the same (please don't change that)
        //         and it is easier to retrieve the id from the 3d discretization.
        //         Then, we use the id to extract the node from the 2d mesh.
        switch (dims.size())
        {
          case 2:   // node_scalar
            scalar_field = metaData.get_field<SFT> (stk::topology::NODE_RANK, stateName);
            TEUCHOS_TEST_FOR_EXCEPTION (scalar_field==0, std::runtime_error, "Error! Field not found.\n");
            for (size_t node=0; node<dims[1]; ++node)
            {
              nodeId3d = ElNodeID[cell][sideNodes(side,node)];
              stk::mesh::EntityKey key(stk::topology::NODE_RANK, nodeId3d+1);
              e = bulkData.get_entity(key);
              values = stk::mesh::field_data(*scalar_field, e);
              values[0] = field(sideSet_idx,node);
            }
            break;
          case 3:   // node_vector
            vector_field = metaData.get_field<VFT> (stk::topology::NODE_RANK, stateName);
            TEUCHOS_TEST_FOR_EXCEPTION (vector_field==0, std::runtime_error, "Error! Field not found.\n");
            for (size_t node=0; node<dims[1]; ++node)
            {
              nodeId3d = ElNodeID[cell][sideNodes(side,node)];
              e = bulkData.get_entity(stk::topology::NODE_RANK, nodeId3d+1);
              values = stk::mesh::field_data(*vector_field, e);
              for (unsigned int i=0; i<dims[2]; ++i)
                values[i] = field(sideSet_idx,node,i);
            }
            break;
          default:  // error!
            TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Unexpected field dimension (only node_scalar/node_vector for now).\n");
        }
      }
    } else {
      for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
      {
        // Get the data that corresponds to the side
        const int cell = sideSet.elem_LID(sideSet_idx);
        const int side = sideSet.side_local_id(sideSet_idx);

        // Notice: in the following, we retrieve the id of the stk node using the 3d mesh.
        //         This is because the id of entities is the same (please don't change that)
        //         and it is easier to retrieve the id from the 3d discretization.
        //         Then, we use the id to extract the node from the 2d mesh.
        switch (dims.size())
        {
          case 3:   // node_scalar
            scalar_field = metaData.get_field<SFT> (stk::topology::NODE_RANK, stateName);
            TEUCHOS_TEST_FOR_EXCEPTION (scalar_field==0, std::runtime_error, "Error! Field not found.\n");
            for (size_t node=0; node<dims[2]; ++node)
            {
              nodeId3d = ElNodeID[cell][sideNodes(side,node)];
              stk::mesh::EntityKey key(stk::topology::NODE_RANK, nodeId3d+1);
              e = bulkData.get_entity(key);
              values = stk::mesh::field_data(*scalar_field, e);
              values[0] = field(cell,side,node);
            }
            break;
          case 4:   // node_vector
            vector_field = metaData.get_field<VFT> (stk::topology::NODE_RANK, stateName);
            TEUCHOS_TEST_FOR_EXCEPTION (vector_field==0, std::runtime_error, "Error! Field not found.\n");
            for (size_t node=0; node<dims[2]; ++node)
            {
              nodeId3d = ElNodeID[cell][sideNodes(side,node)];
              e = bulkData.get_entity(stk::topology::NODE_RANK, nodeId3d+1);
              values = stk::mesh::field_data(*vector_field, e);
              for (unsigned int i=0; i<dims[3]; ++i)
                values[i] = field(cell,side,node,i);
            }
            break;
          default:  // error!
            TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Unexpected field dimension (only node_scalar/node_vector for now).\n");
        }
      }
    }
  }
  else
  {
    // It is a layered mesh, with column-wise ordering. This means that the GID of the node in the 2D mesh
    // will not coincide with the GID of the node in the 3D mesh.

    GO nodeId2d;

    // Loop on the sides of this sideSet that are in this workset
    sideSet = workset.sideSetViews->at(sideSetName);
    if (useCollapsedSidesets) { 
      for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
      {
        // Get the data that corresponds to the side
        const int cell = sideSet.elem_LID(sideSet_idx);
        const int side = sideSet.side_local_id(sideSet_idx);

        switch (dims.size())
        {
          case 2:   // node_scalar
            scalar_field = metaData.get_field<SFT> (stk::topology::NODE_RANK, stateName);
            TEUCHOS_TEST_FOR_EXCEPTION (scalar_field==0, std::runtime_error, "Error! Field not found.\n");
            for (size_t node=0; node<dims[1]; ++node)
            {
              nodeId3d = ElNodeID[cell][sideNodes(side,node)];
              nodeId2d = layeredMeshNumbering->getColumnId(nodeId3d);
              stk::mesh::EntityKey key(stk::topology::NODE_RANK, nodeId2d+1);
              e = bulkData.get_entity(key);
              values = stk::mesh::field_data(*scalar_field, e);
              values[0] = field(sideSet_idx,node);
            }
            break;
          case 3:   // node_vector
            vector_field = metaData.get_field<VFT> (stk::topology::NODE_RANK, stateName);
            TEUCHOS_TEST_FOR_EXCEPTION (vector_field==0, std::runtime_error, "Error! Field not found.\n");
            for (size_t node=0; node<dims[1]; ++node)
            {
              nodeId3d = ElNodeID[cell][sideNodes(side,node)];
              nodeId2d = layeredMeshNumbering->getColumnId(nodeId3d);
              stk::mesh::EntityKey key(stk::topology::NODE_RANK, nodeId2d+1);
              e = bulkData.get_entity(key);
              values = stk::mesh::field_data(*vector_field, e);
              for (size_t i=0; i<dims[2]; ++i)
                values[i] = field(sideSet_idx,node,i);
            }
            break;
          default:  // error!
            TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Unexpected field dimension (only node_scalar/node_vector for now).\n");
        }
      }
    } else {
      for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
      {
        // Get the data that corresponds to the side
        const int cell = sideSet.elem_LID(sideSet_idx);
        const int side = sideSet.side_local_id(sideSet_idx);

        switch (dims.size())
        {
          case 3:   // node_scalar
            scalar_field = metaData.get_field<SFT> (stk::topology::NODE_RANK, stateName);
            TEUCHOS_TEST_FOR_EXCEPTION (scalar_field==0, std::runtime_error, "Error! Field not found.\n");
            for (size_t node=0; node<dims[2]; ++node)
            {
              nodeId3d = ElNodeID[cell][sideNodes(side,node)];
              nodeId2d = layeredMeshNumbering->getColumnId(nodeId3d);
              stk::mesh::EntityKey key(stk::topology::NODE_RANK, nodeId2d+1);
              e = bulkData.get_entity(key);
              values = stk::mesh::field_data(*scalar_field, e);
              values[0] = field(cell,side,node);
            }
            break;
          case 4:   // node_vector
            vector_field = metaData.get_field<VFT> (stk::topology::NODE_RANK, stateName);
            TEUCHOS_TEST_FOR_EXCEPTION (vector_field==0, std::runtime_error, "Error! Field not found.\n");
            for (size_t node=0; node<dims[2]; ++node)
            {
              nodeId3d = ElNodeID[cell][sideNodes(side,node)];
              nodeId2d = layeredMeshNumbering->getColumnId(nodeId3d);
              stk::mesh::EntityKey key(stk::topology::NODE_RANK, nodeId2d+1);
              e = bulkData.get_entity(key);
              values = stk::mesh::field_data(*vector_field, e);
              for (size_t i=0; i<dims[3]; ++i)
                values[i] = field(cell,side,node,i);
            }
            break;
          default:  // error!
            TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! Unexpected field dimension (only node_scalar/node_vector for now).\n");
        }
      }
    }

  }
}

} // Namespace PHAL
