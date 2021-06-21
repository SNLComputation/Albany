//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_ADD_NOISE_HPP
#define PHAL_ADD_NOISE_HPP

#include <Albany_Layouts.hpp>
#include <Albany_ScalarOrdinalTypes.hpp>

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include <Teuchos_ParameterList.hpp>

#include <memory> // std::unique_ptr
#include <random>

namespace PHAL
{

/** \brief Field Norm Evaluator

    This evaluator evaluates the norm of a field
*/

template<typename EvalT, typename Traits, typename ScalarT>
class AddNoiseBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                     public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  AddNoiseBase (const Teuchos::ParameterList& p);
  AddNoiseBase (const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void preEvaluate(typename Traits::PreEvalData d);

  void evaluateFields(typename Traits::EvalData d);

private:
  void setup(const std::string& fieldName,
             const std::string& noisyFieldName,
             const Teuchos::RCP<PHX::DataLayout>& layout,
             const Teuchos::ParameterList& pdf_params);

  // Input:
  PHX::MDField<const ScalarT> field;

  // Output:
  PHX::MDField<ScalarT> noisy_field;

  enum PDFType {UNIFORM, NORMAL};
  PDFType pdf_type;

  std::default_random_engine                              generator;
  std::unique_ptr<std::uniform_real_distribution<double>> pdf_uniform;
  std::unique_ptr<std::normal_distribution<double>>       pdf_normal;

  double rel_noise;
  double abs_noise;

  int seed;
  bool reset_seed_pre_eval;
  bool noise_free;
};

// Some shortcut names
template<typename EvalT, typename Traits>
using AddNoiseST = AddNoiseBase<EvalT,Traits,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using AddNoiseMST = AddNoiseBase<EvalT,Traits,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using AddNoisePST = AddNoiseBase<EvalT,Traits,typename EvalT::ParamScalarT>;

template<typename EvalT, typename Traits>
using AddNoiseRT = AddNoiseBase<EvalT,Traits,RealType>;
} // Namespace PHAL

#endif // PHAL_ADD_NOISE_HPP
