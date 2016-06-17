//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_PROBLEMUTILS_HPP
#define ALBANY_PROBLEMUTILS_HPP

#include <vector>
#include <string>

#include "Teuchos_RCP.hpp"
#include "Teuchos_VerboseObject.hpp"

#include "Albany_Layouts.hpp"

#include "Intrepid2_Basis.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"

#include "Kokkos_DynRankView.hpp"


namespace Albany {

  //! Helper Factory function to construct Intrepid2 Basis from Shards CellTopologyData
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> >
  getIntrepid2Basis(const CellTopologyData& ctd, bool compositeTet=false);
}

#endif 
