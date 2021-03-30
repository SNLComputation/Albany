//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_BLOCKED_DISCRETIZATION_HPP
#define ALBANY_BLOCKED_DISCRETIZATION_HPP

#include <utility>
#include <vector>

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_DataTypes.hpp"
#include "utility/Albany_ThyraBlockedCrsMatrixFactory.hpp"
#include "utility/Albany_ThyraUtils.hpp"

#include "Panzer_BlockedDOFManager.hpp"
//#include "Albany_NullSpaceUtils.hpp"

namespace Albany
{

  // This BlockedSTKDiscretization class implements multiple discretization blocks and serves them up as Thyra_ProductVectors.
  // In this implementation, each block has to be the same, and the discretization for each block is defined here

  class BlockedSTKDiscretization : public AbstractDiscretization
  {
  public:
    // The type of discretization stored internally. For now, only STK.
    // TODO: generalize
    using disc_type = STKDiscretization;

    //! Constructor
    BlockedSTKDiscretization(
        const Teuchos::RCP<Teuchos::ParameterList> &discParams,
        Teuchos::RCP<AbstractSTKMeshStruct> &stkMeshStruct,
        const Teuchos::RCP<const Teuchos_Comm> &comm,
        const Teuchos::RCP<RigidBodyModes> &rigidBodyModes = Teuchos::null,
        const std::map<int, std::vector<std::string>> &sideSetEquations =
            std::map<int, std::vector<std::string>>());

    BlockedSTKDiscretization(
        const Teuchos::RCP<Teuchos::ParameterList> &discParams,
        Teuchos::Array<Teuchos::RCP<AbstractSTKMeshStruct>> &stkMeshStruct,
        const Teuchos::RCP<const Teuchos_Comm> &comm,
        const Teuchos::RCP<RigidBodyModes> &rigidBodyModes = Teuchos::null,
        const std::map<int, std::vector<std::string>> &sideSetEquations =
            std::map<int, std::vector<std::string>>());

    //! Destructor
    virtual ~BlockedSTKDiscretization() = default;

    /** Does a fieldOrder string require blocking? 
    * A field order is basically stetup like this
    *    blocked: <field 0> <field 1> 
    * where two blocks will be created. To merge fields
    * between blocks use a hyphen, i.e.
    *    blocked: <field 0> <field 1> - <field 2> - <field 3>
    * This will create 2 blocks, the first contains only <field 0>
    * and the second combines <field 1>, <field 2> and <field 3>. Note
    * the spaces before and after the hyphen, these are important!
    */

    static bool requiresBlocking(const std::string &fieldorder);

    static void buildBlocking(const std::string &fieldorder, std::vector<std::vector<std::string>> &blocks);

    static void buildNewBlocking(const std::string &fieldorder, std::vector<std::vector<std::string>> &blocks);

    int numBlocks () const { return m_blocks.size(); }

    Teuchos::RCP<disc_type> getBlock (const int iblock) { return m_blocks[iblock]; }

    void printConnectivity() const;
    void printConnectivity(const size_t i_block) const;

    int getBlockFADLength(const size_t i_block)
    {
      return m_blocks[i_block]->getFADLength();
    }

    int getBlockFADOffset(const size_t i_block)
    {
      int offset = 0;
      for (size_t ii_block = 0; ii_block < i_block; ++ii_block)
        offset += getBlockFADLength(ii_block);

      return offset;
    }

    //! Get node vector space (owned and overlapped)
    Teuchos::RCP<const Thyra_VectorSpace> getNodeVectorSpace() const;
    Teuchos::RCP<const Thyra_VectorSpace> getNodeVectorSpace(const size_t i_block) const;
    Teuchos::RCP<const Thyra_VectorSpace> getOverlapNodeVectorSpace() const;
    Teuchos::RCP<const Thyra_VectorSpace> getOverlapNodeVectorSpace(const size_t i_block) const;

    //! Get solution DOF vector space (owned and overlapped).
    Teuchos::RCP<const Thyra_VectorSpace> getVectorSpace() const;
    Teuchos::RCP<const Thyra_VectorSpace> getVectorSpace(const size_t i_block) const;
    Teuchos::RCP<const Thyra_VectorSpace> getOverlapVectorSpace() const;
    Teuchos::RCP<const Thyra_VectorSpace> getOverlapVectorSpace(const size_t i_block) const;

    //! Get Field node vector space (owned and overlapped)
    Teuchos::RCP<const Thyra_VectorSpace> getNodeVectorSpace(const std::string &field_name) const;
    Teuchos::RCP<const Thyra_VectorSpace> getNodeVectorSpace(const size_t i_block, const std::string &field_name) const;
    Teuchos::RCP<const Thyra_VectorSpace> getOverlapNodeVectorSpace(const std::string &field_name) const;
    Teuchos::RCP<const Thyra_VectorSpace> getOverlapNodeVectorSpace(const size_t i_block, const std::string &field_name) const;

    //! Get Field vector space (owned and overlapped)
    Teuchos::RCP<const Thyra_VectorSpace> getVectorSpace(const std::string &field_name) const;
    Teuchos::RCP<const Thyra_VectorSpace> getVectorSpace(const size_t i_block, const std::string &field_name) const;
    Teuchos::RCP<const Thyra_VectorSpace> getOverlapVectorSpace(const std::string &field_name) const;
    Teuchos::RCP<const Thyra_VectorSpace> getOverlapVectorSpace(const size_t i_block, const std::string &field_name) const;

    //! Create a Jacobian operator (owned and overlapped)
    Teuchos::RCP<Thyra_LinearOp>
    createJacobianOp() const
    {
      return m_jac_factory->createOp();
    }
    Teuchos::RCP<Thyra_LinearOp>
    createOverlapJacobianOp() const
    {
      return m_overlap_jac_factory->createOp();
    }

    //! Get node vector space (owned and overlapped)
    Teuchos::RCP<const Thyra_ProductVectorSpace>
    getNodeProductVectorSpace() const
    {
      return m_node_pvs;
    }
    Teuchos::RCP<const Thyra_ProductVectorSpace>
    getOverlapNodeProductVectorSpace() const
    {
      return m_overlap_node_pvs;
    }

    //! Get solution DOF vector space (owned and overlapped).
    Teuchos::RCP<const Thyra_ProductVectorSpace>
    getProductVectorSpace() const
    {
      return m_pvs;
    }

    Teuchos::RCP<const Thyra_ProductVectorSpace>
    getOverlapProductVectorSpace() const
    {
      return m_overlap_pvs;
    }

    void computeProductVectorSpaces();
    void computeGraphs();
    void computeGraphs(const size_t i_block, const size_t j_block);

    //! Get Field node vector space (owned and overlapped)
    Teuchos::RCP<const Thyra_ProductVectorSpace>
    getNodeProductVectorSpace(const std::string &field_name) const;
    Teuchos::RCP<const Thyra_ProductVectorSpace>
    getOverlapNodeProductVectorSpace(const std::string &field_name) const;

    //! Get Field vector space (owned and overlapped)
    Teuchos::RCP<const Thyra_ProductVectorSpace>
    getProductVectorSpace(const std::string &field_name) const;
    Teuchos::RCP<const Thyra_ProductVectorSpace>
    getOverlapProductVectorSpace(const std::string &field_name) const;

    //! Create a Jacobian operator (owned and overlapped)
    Teuchos::RCP<Thyra_BlockedLinearOp>
    createBlockedJacobianOp() const
    {
      return m_jac_factory->createOp();
    }

    Teuchos::RCP<Thyra_BlockedLinearOp>
    createOverlapBlockedJacobianOp() const
    {
      return m_overlap_jac_factory->createOp();
    }

    bool
    isExplicitScheme() const
    {
      return false;
    }

    //! Get Node set lists (typedef in Albany_AbstractDiscretization.hpp)
    const NodeSetList &
    getNodeSets() const
    {
      return nodeSets;
    }
    const NodeSetGIDsList &
    getNodeSetGIDs() const
    {
      return nodeSetGIDs;
    }
    const NodeSetCoordList &
    getNodeSetCoords() const
    {
      return nodeSetCoords;
    }

    //! Get Side set lists (typedef in Albany_AbstractDiscretization.hpp)
    const SideSetList &
    getSideSets(const int workset) const
    {
      return this->getSideSets(0, workset);
    }
    const SideSetList &
    getSideSets(const size_t i_block, const int workset) const
    {
      return m_blocks[i_block]->getSideSets(workset);
    }

    //! Get Side set lists (typedef in Albany_AbstractDiscretization.hpp)
    const LocalSideSetInfoList &
    getSideSetViews(const int workset) const
    {
      return this->getSideSetViews(0, workset);
    }
    const LocalSideSetInfoList &
    getSideSetViews(const size_t i_block, const int workset) const
    {
      return m_blocks[i_block]->getSideSetViews(workset);
    }

    //! Get local DOF views for GatherVerticallyContractedSolution
    const std::map<std::string, Kokkos::View<LO ****, PHX::Device>> &
    getLocalDOFViews(const int workset) const
    {
      return this->getLocalDOFViews(0, workset);
    }
    const std::map<std::string, Kokkos::View<LO ****, PHX::Device>> &
    getLocalDOFViews(const size_t i_block, const int workset) const
    {
      return m_blocks[i_block]->getLocalDOFViews(workset);
    }

    //! Get connectivity map from elementGID to workset
    WsLIDList &
    getElemGIDws()
    {
      return elemGIDws;
    }
    WsLIDList const &
    getElemGIDws() const
    {
      return elemGIDws;
    }

    //! Get map from ws, elem, node [, eq] -> [Node|DOF] GID
    const Conn &
    getWsElNodeEqID() const
    {
      return wsElNodeEqID;
    }

    const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO>>>::type &
    getWsElNodeID() const
    {
      return wsElNodeID;
    }

    //! Get IDArray for (Ws, Local Node, nComps) -> (local) NodeLID, works for
    //! both scalar and vector fields
    const std::vector<IDArray> &
    getElNodeEqID(const std::string &field_name) const
    {
      return this->getElNodeEqID(0, field_name);
    }
    const std::vector<IDArray> &
    getElNodeEqID(const size_t i_block, const std::string &field_name) const
    {
      return m_blocks[i_block]->getElNodeEqID(field_name);
    }

    void
    setBlockedDOFManager(Teuchos::RCP<panzer::BlockedDOFManager> blockedDOFManager_)
    {
      blockedDOFManager = blockedDOFManager_;
    }
    Teuchos::RCP<panzer::BlockedDOFManager>
    getBlockedDOFManager()
    {
      return blockedDOFManager;
    }

    const NodalDOFManager &
    getDOFManager(const std::string &field_name) const
    {
      return this->getDOFManager(0, field_name);
    }
    const NodalDOFManager &
    getDOFManager(const size_t i_block, const std::string &field_name) const
    {
      return m_blocks[i_block]->getDOFManager(field_name);
    }

    const NodalDOFManager &
    getOverlapDOFManager(const std::string &field_name) const
    {
      return this->getOverlapDOFManager(0, field_name);
    }
    const NodalDOFManager &
    getOverlapDOFManager(const size_t i_block, const std::string &field_name) const
    {
      return m_blocks[i_block]->getOverlapDOFManager(field_name);
    }

    const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double *>>>::type &
    getCoords() const
    {
      return coords;
    }

    //! Print the coordinates for debugging
    void printCoords() const;
    void printCoords(const size_t i_block) const;

    //! Set stateArrays
    void
    setStateArrays(StateArrays &sa)
    {
      stateArrays = sa;
    }

    //! Get stateArrays
    StateArrays &
    getStateArrays()
    {
      return stateArrays;
    }

    //! Get nodal parameters state info struct
    const StateInfoStruct &
    getNodalParameterSIS() const
    {
      return stkMeshStruct->getFieldContainer()->getNodalParameterSIS();
    }

    //! Retrieve Vector (length num worksets) of element block names
    const WorksetArray<std::string>::type &
    getWsEBNames() const
    {
      return wsEBNames;
    }
    //! Retrieve Vector (length num worksets) of physics set index
    const WorksetArray<int>::type &
    getWsPhysIndex() const
    {
      return wsPhysIndex;
    }

    // Retrieve mesh struct
    Teuchos::RCP<AbstractSTKMeshStruct>
    getSTKMeshStruct() const
    {
      return stkMeshStruct;
    }
    Teuchos::RCP<AbstractMeshStruct>
    getMeshStruct() const
    {
      return stkMeshStruct;
    }

    const SideSetDiscretizationsType &
    getSideSetDiscretizations() const
    {
      return sideSetDiscretizations;
    }

    const std::map<std::string, std::map<GO, GO>> &
    getSideToSideSetCellMap() const
    {
      return sideToSideSetCellMap;
    }

    const std::map<std::string, std::map<GO, std::vector<int>>> &
    getSideNodeNumerationMap() const
    {
      return sideNodeNumerationMap;
    }

    //! Flag if solution has a restart values -- used in Init Cond
    bool
    hasRestartSolution() const
    {
      return stkMeshStruct->hasRestartSolution();
    }

    //! If restarting, convenience function to return restart data time
    double
    restartDataTime() const
    {
      return stkMeshStruct->restartDataTime();
    }

    //! After mesh modification, need to update the element connectivity and nodal
    //! coordinates
    void updateMesh();
    void updateMesh(const size_t i_block);

    //! Function that transforms an STK mesh of a unit cube (for LandIce problems)
    void transformMesh();
    void transformMesh(const size_t i_block);

    //! Get number of spatial dimensions
    int
    getNumDim() const
    {
      return stkMeshStruct->numDim;
    }

    //! Get number of total DOFs per node
    int
    getNumEq() const
    {
      return this->getNumEq(0);
    }
    int
    getNumEq(const size_t i_block) const
    {
      return m_blocks[i_block]->getNumEq();
    }

    Teuchos::RCP<LayeredMeshNumbering<GO>>
    getLayeredMeshNumbering() const
    {
      return stkMeshStruct->layered_mesh_numbering;
    }

    const stk::mesh::MetaData &
    getSTKMetaData() const
    {
      return this->getSTKMetaData(0);
    }
    const stk::mesh::MetaData &
    getSTKMetaData(const size_t i_block) const
    {
      return m_blocks[i_block]->getSTKMetaData();
    }
    const stk::mesh::BulkData &
    getSTKBulkData() const
    {
      return this->getSTKBulkData(0);
    }
    const stk::mesh::BulkData &
    getSTKBulkData(const size_t i_block) const
    {
      return m_blocks[i_block]->getSTKBulkData();
    }

    // --- Get/set solution/residual/field vectors to/from mesh --- //

    Teuchos::RCP<Thyra_MultiVector>
    getBlockedSolutionField(const bool overlapped = false) const;
    // These are identical for a blocked discretization, keep the below to keep the interface the same
    Teuchos::RCP<Thyra_MultiVector>
    getBlockedSolutionMV(const bool overlapped = false) const;

    // GAH - These all need a serious look, they are just stubbed in to get things compiling

    // Used very often, so make it a function
    GO stk_gid(const size_t i_block, const stk::mesh::Entity e) const
    {
      // STK numbering is 1-based, while we want 0-based.
      return this->getSTKBulkData(i_block).identifier(e) - 1;
    }

    //! Retrieve coordinate vector (num_used_nodes * 3)
    const Teuchos::ArrayRCP<double> &
    getCoordinates() const
    {
      return this->getCoordinates(0);
    }
    const Teuchos::ArrayRCP<double> &
    getCoordinates(const size_t i_block) const
    {
      return m_blocks[i_block]->getCoordinates();
    }

    Teuchos::RCP<Thyra_Vector>
    getSolutionField(bool overlapped) const
    {
      return this->getSolutionField(0, overlapped);
    }
    Teuchos::RCP<Thyra_Vector>
    getSolutionField(const size_t i_block, bool overlapped) const
    {
      return m_blocks[i_block]->getSolutionField(overlapped);
    }

    Teuchos::RCP<Thyra_MultiVector>
    getSolutionMV(bool overlapped) const
    {
      return this->getSolutionMV(0, overlapped);
    }
    Teuchos::RCP<Thyra_MultiVector>
    getSolutionMV(const size_t i_block, bool overlapped) const
    {
      return m_blocks[i_block]->getSolutionMV(overlapped);
    }

    void
    getField(Thyra_Vector &result, const std::string &name) const
    {
      this->getField(0, result, name);
    }
    void
    getField(const size_t i_block, Thyra_Vector &result, const std::string &name) const
    {
      m_blocks[i_block]->getField(result, name);
    }

    void
    getSolutionField(Thyra_Vector &result, const bool overlapped) const
    {
      this->getSolutionField(0, result, overlapped);
    }
    void
    getSolutionField(const size_t i_block, Thyra_Vector &result, const bool overlapped) const
    {
      m_blocks[i_block]->getSolutionField(result, overlapped);
    }

    void
    setField(
        const Thyra_Vector &result,
        const std::string &name,
        bool overlapped)
    {
      this->setField(0, result, name, overlapped);
    }
    void
    setField(
        const size_t i_block,
        const Thyra_Vector &result,
        const std::string &name,
        bool overlapped)
    {
      m_blocks[i_block]->setField(result, name, overlapped);
    }

    void
    writeSolution(
        const Thyra_Vector &soln,
        const Teuchos::RCP<const Thyra_MultiVector> &soln_dxdp,
        const double time,
        const bool overlapped)
    {
      this->writeSolution(0, soln, soln_dxdp, time, overlapped);
    }
    void
    writeSolution(
        const size_t i_block,
        const Thyra_Vector &soln,
        const Teuchos::RCP<const Thyra_MultiVector> &soln_dxdp,
        const double time,
        const bool overlapped)
    {
      m_blocks[i_block]->writeSolution(soln, soln_dxdp, time, overlapped);
    }

    void
    writeSolution(
        const Thyra_Vector &soln,
        const Teuchos::RCP<const Thyra_MultiVector> &soln_dxdp,
        const Thyra_Vector &soln_dot,
        const double time,
        const bool overlapped)
    {
      this->writeSolution(0, soln, soln_dxdp, soln_dot, time, overlapped);
    }
    void
    writeSolution(
        const size_t i_block,
        const Thyra_Vector &soln,
        const Teuchos::RCP<const Thyra_MultiVector> &soln_dxdp,
        const Thyra_Vector &soln_dot,
        const double time,
        const bool overlapped)
    {
      m_blocks[i_block]->writeSolution(soln, soln_dxdp, soln_dot, time, overlapped);
    }

    void
    writeSolution(
        const Thyra_Vector &soln,
        const Teuchos::RCP<const Thyra_MultiVector> &soln_dxdp,
        const Thyra_Vector &soln_dot,
        const Thyra_Vector &soln_dotdot,
        const double time,
        const bool overlapped)
    {
      this->writeSolution(0, soln, soln_dxdp, soln_dot, soln_dotdot, time, overlapped);
    }
    void
    writeSolution(
        const size_t i_block,
        const Thyra_Vector &soln,
        const Teuchos::RCP<const Thyra_MultiVector> &soln_dxdp,
        const Thyra_Vector &soln_dot,
        const Thyra_Vector &soln_dotdot,
        const double time,
        const bool overlapped)
    {
      m_blocks[i_block]->writeSolution(soln, soln_dxdp, soln_dot, soln_dotdot, time, overlapped);
    }

    void
    writeSolutionMV(
        const Thyra_MultiVector &soln,
        const Teuchos::RCP<const Thyra_MultiVector> &soln_dxdp,
        const double time,
        const bool overlapped)
    {
      this->writeSolutionMV(0, soln, soln_dxdp, time, overlapped);
    }
    void
    writeSolutionMV(
        const size_t i_block,
        const Thyra_MultiVector &soln,
        const Teuchos::RCP<const Thyra_MultiVector> &soln_dxdp,
        const double time,
        const bool overlapped)
    {
      m_blocks[i_block]->writeSolutionMV(soln, soln_dxdp, time, overlapped);
    }

    void
    writeSolutionToMeshDatabase(
        const Thyra_Vector &soln,
        const Teuchos::RCP<const Thyra_MultiVector> &soln_dxdp,
        const double /* time */,
        const bool overlapped)
    {
      this->writeSolutionToMeshDatabase(0, soln, soln_dxdp, 0.0, overlapped);
    }
    void
    writeSolutionToMeshDatabase(
        const size_t i_block,
        const Thyra_Vector &soln,
        const Teuchos::RCP<const Thyra_MultiVector> &soln_dxdp,
        const double /* time */,
        const bool overlapped)
    {
      m_blocks[i_block]->writeSolutionToMeshDatabase(soln, soln_dxdp, 0.0, overlapped);
    }

    void
    writeSolutionToMeshDatabase(
        const Thyra_Vector &soln,
        const Teuchos::RCP<const Thyra_MultiVector> &soln_dxdp,
        const Thyra_Vector &soln_dot,
        const double /* time */,
        const bool overlapped)
    {
      this->writeSolutionToMeshDatabase(soln, soln_dxdp, soln_dot, 0.0, overlapped);
    }
    void
    writeSolutionToMeshDatabase(
        const size_t i_block,
        const Thyra_Vector &soln,
        const Teuchos::RCP<const Thyra_MultiVector> &soln_dxdp,
        const Thyra_Vector &soln_dot,
        const double /* time */,
        const bool overlapped)
    {
      m_blocks[i_block]->writeSolutionToMeshDatabase(soln, soln_dxdp, soln_dot, 0.0, overlapped);
    }

    void
    writeSolutionToMeshDatabase(
        const Thyra_Vector &soln,
        const Teuchos::RCP<const Thyra_MultiVector> &soln_dxdp,
        const Thyra_Vector &soln_dot,
        const Thyra_Vector &soln_dotdot,
        const double /* time */,
        const bool overlapped)
    {
      this->writeSolutionToMeshDatabase(0, soln, soln_dxdp, soln_dot, soln_dotdot, 0.0, overlapped);
    }
    void
    writeSolutionToMeshDatabase(
        const size_t i_block,
        const Thyra_Vector &soln,
        const Teuchos::RCP<const Thyra_MultiVector> &soln_dxdp,
        const Thyra_Vector &soln_dot,
        const Thyra_Vector &soln_dotdot,
        const double /* time */,
        const bool overlapped)
    {
      m_blocks[i_block]->writeSolutionToMeshDatabase(soln, soln_dxdp, soln_dot, soln_dotdot, 0.0, overlapped);
    }

    void
    writeSolutionMVToMeshDatabase(
        const Thyra_MultiVector &soln,
        const Teuchos::RCP<const Thyra_MultiVector> &soln_dxdp,
        const double /* time */,
        const bool overlapped)
    {
      this->writeSolutionMVToMeshDatabase(soln, soln_dxdp, 0.0, overlapped);
    }
    void
    writeSolutionMVToMeshDatabase(
        const size_t i_block,
        const Thyra_MultiVector &soln,
        const Teuchos::RCP<const Thyra_MultiVector> &soln_dxdp,
        const double /* time */,
        const bool overlapped)
    {
      m_blocks[i_block]->writeSolutionMVToMeshDatabase(soln, soln_dxdp, 0.0, overlapped);
    }

    void
    writeSolutionToFile(
        const Thyra_Vector &soln,
        const double time,
        const bool overlapped)
    {
      this->writeSolutionToFile(0, soln, time, overlapped);
    }
    void
    writeSolutionToFile(
        const size_t i_block,
        const Thyra_Vector &soln,
        const double time,
        const bool overlapped)
    {
      m_blocks[i_block]->writeSolutionToFile(soln, time, overlapped);
    }

    void
    writeSolutionMVToFile(
        const Thyra_MultiVector &soln,
        const double time,
        const bool overlapped)
    {
      this->writeSolutionMVToFile(0, soln, time, overlapped);
    }
    void
    writeSolutionMVToFile(
        const size_t i_block,
        const Thyra_MultiVector &soln,
        const double time,
        const bool overlapped)
    {
      m_blocks[i_block]->writeSolutionMVToFile(soln, time, overlapped);
    }

    Teuchos::RCP<const GlobalLocalIndexer>
    getGlobalLocalIndexer(const std::string &field_name) const
    {
      return this->getGlobalLocalIndexer(0, field_name);
    }
    Teuchos::RCP<const GlobalLocalIndexer>
    getGlobalLocalIndexer(const size_t i_block, const std::string &field_name) const
    {
      return m_blocks[i_block]->getGlobalLocalIndexer(field_name);
    }

    Teuchos::RCP<const GlobalLocalIndexer>
    getOverlapGlobalLocalIndexer(const std::string &field_name) const
    {
      return this->getOverlapGlobalLocalIndexer(0, field_name);
    }
    Teuchos::RCP<const GlobalLocalIndexer>
    getOverlapGlobalLocalIndexer(const size_t i_block, const std::string &field_name) const
    {
      return m_blocks[i_block]->getOverlapGlobalLocalIndexer(field_name);
    }

    // GAH - End serious look

    // ==================== Members =================== //

    Teuchos::RCP<Teuchos::FancyOStream> out;

    //! Teuchos communicator
    Teuchos::RCP<const Teuchos_Comm> comm;

    //! Unknown map and node map
    Teuchos::RCP<const Thyra_ProductVectorSpace> m_pvs;
    Teuchos::RCP<const Thyra_ProductVectorSpace> m_node_pvs;

    //! Overlapped unknown map and node map
    Teuchos::RCP<const Thyra_ProductVectorSpace> m_overlap_pvs;
    Teuchos::RCP<const Thyra_ProductVectorSpace> m_overlap_node_pvs;

    //! Jacobian matrix graph proxy (owned and overlap)
    Teuchos::RCP<ThyraBlockedCrsMatrixFactory> m_jac_factory;
    Teuchos::RCP<ThyraBlockedCrsMatrixFactory> m_overlap_jac_factory;

    //! Equations that are defined only on some side sets of the mesh
    std::map<int, std::vector<std::string>> sideSetEquations;

    //! Number of elements on this processor
    unsigned int numMyElements;

    //! node sets stored as std::map(string ID, int vector of GIDs)
    NodeSetList nodeSets;
    NodeSetGIDsList nodeSetGIDs;
    NodeSetCoordList nodeSetCoords;

    //! Connectivity array [workset, element, local-node, Eq] => LID
    Conn wsElNodeEqID;

    //! Connectivity array [workset, element, local-node] => GID
    WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO>>>::type wsElNodeID;

    mutable Teuchos::ArrayRCP<double> coordinates;
    Teuchos::RCP<Thyra_MultiVector> coordMV;
    WorksetArray<std::string>::type wsEBNames;
    WorksetArray<int>::type wsPhysIndex;
    WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double *>>>::type coords;
    WorksetArray<Teuchos::ArrayRCP<double>>::type sphereVolume;
    WorksetArray<Teuchos::ArrayRCP<double *>>::type latticeOrientation;

    //! Connectivity map from elementGID to workset and LID in workset
    WsLIDList elemGIDws;

    // States: vector of length worksets of a map from field name to shards array
    StateArrays stateArrays;
    std::vector<std::vector<std::vector<double>>> nodesOnElemStateVec;

    //! list of all owned nodes, saved for setting solution
    std::vector<stk::mesh::Entity> ownednodes;
    std::vector<stk::mesh::Entity> cells;

    //! list of all overlap nodes, saved for getting coordinates for mesh motion
    std::vector<stk::mesh::Entity> overlapnodes;

    //! Number of elements on this processor
    int numOwnedNodes;
    int numOverlapNodes;
    GO maxGlobalNodeGID;

    // Needed to pass coordinates to ML.
    Teuchos::RCP<RigidBodyModes> rigidBodyModes;

    Teuchos::RCP<AbstractSTKMeshStruct> stkMeshStruct;

    Teuchos::RCP<Teuchos::ParameterList> discParams;

    // Sideset discretizations
    std::map<std::string, Teuchos::RCP<AbstractDiscretization>>
        sideSetDiscretizations;
    std::map<std::string, std::map<GO, GO>> sideToSideSetCellMap;
    std::map<std::string, std::map<GO, std::vector<int>>> sideNodeNumerationMap;
    std::map<std::string, Teuchos::RCP<Thyra_BlockedLinearOp>> projectors;
    std::map<std::string, Teuchos::RCP<Thyra_BlockedLinearOp>> ov_projectors;

  private:
    Teuchos::Array<Teuchos::RCP<disc_type>> m_blocks;
    size_t n_m_blocks;

    Teuchos::RCP<ThyraBlockedCrsMatrixFactory> nodalMatrixFactory;

    Teuchos::RCP<panzer::BlockedDOFManager> blockedDOFManager;
  };

} // namespace Albany

#endif // ALBANY_BLOCKED_DISCRETIZATION_HPP
