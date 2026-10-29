// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=lower outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=LOWER-FWD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=lower outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=LOWER-REV
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=upper_batch outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=UPPER-FWD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=upper_batch outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=UPPER-REV

// Preserve native matrix operations through differentiation, including batches.
// LOWER-FWD-LABEL: func.func @lower(
// LOWER-FWD: stablehlo.cholesky
// LOWER-FWD: stablehlo.triangular_solve
// LOWER-FWD: stablehlo.triangular_solve
// LOWER-FWD: stablehlo.dot_general
// LOWER-REV-LABEL: func.func @lower(
// LOWER-REV: stablehlo.cholesky
// LOWER-REV: stablehlo.dot_general
// LOWER-REV: stablehlo.triangular_solve
// LOWER-REV: stablehlo.triangular_solve
func.func @lower(%a: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %l = stablehlo.cholesky %a, lower = true : tensor<2x2xf32>
  return %l : tensor<2x2xf32>
}

// UPPER-FWD-LABEL: func.func @upper_batch(
// UPPER-FWD: stablehlo.cholesky
// UPPER-FWD: stablehlo.triangular_solve
// UPPER-FWD: stablehlo.triangular_solve
// UPPER-FWD: stablehlo.dot_general
// UPPER-REV-LABEL: func.func @upper_batch(
// UPPER-REV: stablehlo.cholesky
// UPPER-REV: stablehlo.dot_general
// UPPER-REV: stablehlo.triangular_solve
// UPPER-REV: stablehlo.triangular_solve
func.func @upper_batch(%a: tensor<2x1x3x3xf64>) -> tensor<2x1x3x3xf64> {
  %u = stablehlo.cholesky %a, lower = false : tensor<2x1x3x3xf64>
  return %u : tensor<2x1x3x3xf64>
}
