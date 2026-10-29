// RUN: not enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" 2>&1 | FileCheck %s
// RUN: not enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" 2>&1 | FileCheck %s

// Complex Cholesky needs conjugation-aware solve rules, rather than real AD.
// CHECK: differentiation requires statically shaped real f32 or f64 tensors
func.func @main(%a: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  %l = stablehlo.cholesky %a, lower = true : tensor<2x2xcomplex<f32>>
  return %l : tensor<2x2xcomplex<f32>>
}
