//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_NULL_SPACE_UTILS_HPP
#define ALBANY_NULL_SPACE_UTILS_HPP

#include "Albany_ThyraTypes.hpp"
#include "Albany_MeshSpecs.hpp"

namespace Albany {

// Forward declaration of a helper class, used to hide Tpetra/Epetra details
struct TraitsImplBase;

class RigidBodyModes {
public:
  //! Construct RBM object.
  RigidBodyModes();

  //! Set sizes of nullspace etc.
  void setParameters(const int numPDEs, const bool computeConstantModes,
      const int physVectorDim = 0, const bool computeRotationModes = false);

  //! Set Piro solver parameter list.
  void setPiroPL(const Teuchos::RCP<Teuchos::ParameterList>& piroParams);

  //! Update the parameter list.
  void updatePL(const Teuchos::RCP<Teuchos::ParameterList>& precParams);

  //! Is ML used on this problem?
  bool isMLUsed() const { return mlUsed; }

  //! Is MueLu used on this problem?
  bool isMueLuUsed() const { return mueLuUsed; }

  //! Is FROSch used on this problem?
  bool isFROSchUsed() const { return froschUsed; }

  //! Pass coordinates and the null space to ML, MueLu or FROSch.
  //! The null space is computed only if
  //! computeConstantModes or computeRotationModes are true
  //! The data accessed through getCoordArrays must have been set
  //! soln_map must be set only if using MueLu or FROSch
  //! Both maps are nonoverlapped.
  void setCoordinatesAndComputeNullspace(
    const Teuchos::RCP<Thyra_MultiVector> &coordMV,
    DiscType interleavedOrdering,
    const Teuchos::RCP<const Thyra_VectorSpace>& soln_vs = Teuchos::null,
    const Teuchos::RCP<const Thyra_VectorSpace>& soln_overlap_vs = Teuchos::null);

  //! Pass only the coordinates.
  void setCoordinates(const Teuchos::RCP<Thyra_MultiVector> &coordMV);

private:
  int numPDEs;
  DiscType interleavedOrdering;
  bool computeConstantModes; //translations
  int physVectorDim;
  bool computeRotationModes;
  int nullSpaceDim;
  bool mlUsed, mueLuUsed, froschUsed, setNonElastRBM;
  bool areProbParametersSet, arePiroParametersSet;

  Teuchos::RCP<Teuchos::ParameterList> plist;

  Teuchos::RCP<Thyra_MultiVector> coordMV;

  Teuchos::RCP<TraitsImplBase> nullSpaceTraits;
};

} // namespace Albany

#endif // ALBANY_NULL_SPACE_UTILS_HPP
