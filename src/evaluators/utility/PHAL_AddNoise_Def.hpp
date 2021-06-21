//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Layouts.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"

#include "PHAL_Utilities.hpp"
#include "PHAL_AddNoise.hpp"

#include <Teuchos_ParameterList.hpp>
#include <stdexcept>
#include <time.h>
#include <random>

//uncomment the following line if you want debug output to be printed to screen
#define OUTPUT_TO_SCREEN

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
AddNoiseBase<EvalT, Traits, ScalarT>::
AddNoiseBase (const Teuchos::ParameterList& p)
{
  std::string fieldName      = p.get<std::string> ("Field Name");
  std::string noisyFieldName = p.get<std::string> ("Noisy Field Name");
  Teuchos::RCP<PHX::DataLayout> layout = p.get<Teuchos::RCP<PHX::DataLayout>>("Field Layout");
  const auto& pdf_params = p.sublist("PDF Parameters");
  
  setup(fieldName,noisyFieldName,layout,pdf_params);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
AddNoiseBase<EvalT, Traits, ScalarT>::
AddNoiseBase (const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl)
{
  std::string fieldName      = p.get<std::string> ("Field Name");
  std::string noisyFieldName = p.get<std::string> ("Noisy Field Name");
  const auto& pdf_params = p.sublist("PDF Parameters");

  const auto& layout_str = p.get<std::string>("Field Layout");
  using FL  = Albany::FieldLocation;
  using FRT = Albany::FieldRankType;
  FL  loc;
  FRT rank;
  
  bool ss_field = false;
  if (layout_str.find("Node")!=std::string::npos) {
    loc = FL::Node;
  } else if (layout_str.find("Cell")!=std::string::npos || 
             layout_str.find("Side")!=std::string::npos) {
    loc = FL::Cell;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
        "Error! Cannot deduce location for field with layout '" + layout_str + "'.\n");
  }

  if (layout_str.find("Side")!=std::string::npos) {
    ss_field = true;
    TEUCHOS_TEST_FOR_EXCEPTION (dl->isSideLayouts, std::runtime_error,
        "Error! Input layouts structure is not that of a side set.\n");
  }

  if (layout_str.find("Scalar")!=std::string::npos) {
    rank = FRT::Scalar;
  } else if (layout_str.find("Vector")!=std::string::npos) {
    rank = FRT::Vector;
  } else if (layout_str.find("Gradient")!=std::string::npos) {
    rank = FRT::Gradient;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
        "Error! Cannot deduce rank for field with layout '" + layout_str + "'.\n");
  }

  Teuchos::RCP<PHX::DataLayout> layout;
  if (ss_field) {
    const auto& ss_name = p.get<std::string>("Side Set Name");
    auto dl_side = dl->side_layouts.at(ss_name);
    layout = Albany::get_field_layout(rank,loc,dl_side);
  } else {
    layout = Albany::get_field_layout(rank,loc,dl);
  }

  setup(fieldName, noisyFieldName,layout,pdf_params);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void AddNoiseBase<EvalT, Traits, ScalarT>::
postRegistrationSetup(typename Traits::SetupData /* d */,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field,fm);
  this->utils.setFieldData(noisy_field,fm);
}

template<typename EvalT, typename Traits, typename ScalarT>
void AddNoiseBase<EvalT, Traits, ScalarT>::
preEvaluate(typename Traits::PreEvalData /* workset */)
{
  // Reset the seed. E.g., each iteration of Newton should solve the system with
  // the same realization of noise.
  if (reset_seed_pre_eval)
    generator.seed (seed);
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void AddNoiseBase<EvalT, Traits, ScalarT>::evaluateFields (typename Traits::EvalData /* workset */)
{
  PHAL::MDFieldIterator<const ScalarT> in(field);
  PHAL::MDFieldIterator<ScalarT> out(noisy_field);

  if (noise_free) {
    for (; !in.done(); ++in, ++out)
      *out = *in;
    return;
  }

  switch (pdf_type) {
    case UNIFORM:
      for (; !in.done(); ++in, ++out)
        *out = abs_noise*(*pdf_uniform)(generator) + (*in)*(1+rel_noise*(*pdf_uniform)(generator));
      break;

    case NORMAL:
      for (; !in.done(); ++in, ++out)
        *out = abs_noise*(*pdf_normal)(generator) + (*in)*(1+rel_noise*(*pdf_normal)(generator));
      break;

    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Error! [PHAL::AddNoiseBase] This exception should never throw.\n");
  }
}

template<typename EvalT, typename Traits, typename ScalarT>
void AddNoiseBase<EvalT, Traits, ScalarT>::
setup(const std::string& fieldName,
      const std::string& noisyFieldName,
      const Teuchos::RCP<PHX::DataLayout>& layout,
      const Teuchos::ParameterList& pdf_params)
{
  noisy_field = decltype(noisy_field)(noisyFieldName, layout);

  field = decltype(field)(fieldName, layout);
  this->addDependentField(field);

  this->addEvaluatedField(noisy_field);

  std::string pdf_type_str = pdf_params.get<std::string>("Noise PDF");
  if (pdf_type_str=="Uniform") {
    pdf_type = UNIFORM;

    double a = pdf_params.get<double>("Lower Bound");
    double b = pdf_params.get<double>("Upper Bound");

    pdf_uniform.reset(new std::uniform_real_distribution<double>(a,b));
  } else if (pdf_type_str=="Normal") {
    pdf_type = NORMAL;

    double mu    = pdf_params.get<double>("Mean");
    double sigma = pdf_params.get<double>("Standard Deviation");

    pdf_normal.reset(new std::normal_distribution<double>(mu,sigma));
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, Teuchos::Exceptions::InvalidParameter, "Error! Invalid noise p.d.f.\n");
  }

  seed = pdf_params.isParameter("Random Seed") ? pdf_params.get<int>("Random Seed") : std::time(nullptr);
  reset_seed_pre_eval = pdf_params.isParameter("Reset Seed With PreEvaluate") ? pdf_params.get<bool>("Reset Seed With PreEvaluate") : true;

  rel_noise = pdf_params.isParameter("Relative Noise") ? pdf_params.get<double>("Relative Noise") :  0.;
  abs_noise = pdf_params.isParameter("Absolute Noise") ? pdf_params.get<double>("Absolute Noise") :  0.;

  TEUCHOS_TEST_FOR_EXCEPTION (rel_noise<0, Teuchos::Exceptions::InvalidParameter, "Error! Relative noise should be non-negative.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (abs_noise<0, Teuchos::Exceptions::InvalidParameter, "Error! Absolute noise should be non-negative.\n");

  noise_free = (rel_noise==0) && (abs_noise==0);

  this->setName("AddNoiseBase(" + noisyFieldName + ")" + PHX::print<EvalT>());
}

} // Namespace PHAL
