//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Moertel_ExplicitTemplateInstantiation.hpp"

#ifdef HAVE_MOERTEL_EXPLICIT_INSTANTIATION
#include "Moertel_NodeT.hpp"
#include "Moertel_NodeT_Def.hpp"

namespace MoertelT {

  MOERTEL_INSTANTIATE_TEMPLATE_CLASS(NodeT)

} // namespace Moertel

// non-member operators at global scope
#ifdef HAVE_MOERTEL_INST_DOUBLE_INT_INT
template std::ostream& operator << (std::ostream& os, const MoertelT::NodeT<3, double, int, int, KokkosNode>& inter);
template std::ostream& operator << (std::ostream& os, const MoertelT::NodeT<2, double, int, int, KokkosNode>& inter);
#endif
#ifdef HAVE_MOERTEL_INST_DOUBLE_INT_LONGLONGINT
template std::ostream& operator << (std::ostream& os, const MoertelT::NodeT<3, double, int, long long, KokkosNode>& inter);
template std::ostream& operator << (std::ostream& os, const MoertelT::NodeT<2, double, int, long long, KokkosNode>& inter);
#endif

#endif
