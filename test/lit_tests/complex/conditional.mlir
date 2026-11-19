// RUN: enzymexlamlir-opt --lower-complex-operations="concat-dimension=first" --canonicalize %s | FileCheck %s --check-prefixes=CHECK,FIRST --implicit-check-not=unrealized_conversion_cast --implicit-check-not='complex<'
// RUN: enzymexlamlir-opt --lower-complex-operations="concat-dimension=last" --canonicalize %s | FileCheck %s --check-prefixes=CHECK,LAST --implicit-check-not=unrealized_conversion_cast --implicit-check-not='complex<'

// CHECK-LABEL: @branches(
// CHECK: "stablehlo.if"(%arg0)
// CHECK: "stablehlo.case"(%arg1)
// FIRST: (tensor<i32>) -> (tensor<2x3xf32>, tensor<i32>)
// LAST:  (tensor<i32>) -> (tensor<3x2xf32>, tensor<i32>)
// FIRST: (tensor<i1>) -> (tensor<2x3xf32>, tensor<i32>)
// LAST:  (tensor<i1>) -> (tensor<3x2xf32>, tensor<i32>)
func.func @branches(%pred: tensor<i1>, %index: tensor<i32>, %re: tensor<3xf32>, %im: tensor<3xf32>) -> (tensor<3xf32>, tensor<i32>) {
  %z = stablehlo.complex %re, %im : tensor<3xcomplex<f32>>
  %w = stablehlo.complex %im, %re : tensor<3xcomplex<f32>>
  %zero = stablehlo.constant dense<0> : tensor<i32>
  %one = stablehlo.constant dense<1> : tensor<i32>
  %out:2 = "stablehlo.if"(%pred) ({
    %pick:2 = "stablehlo.case"(%index) ({
      stablehlo.return %z, %zero : tensor<3xcomplex<f32>>, tensor<i32>
    }, {
      stablehlo.return %w, %one : tensor<3xcomplex<f32>>, tensor<i32>
    }, {
      %sum = stablehlo.add %z, %w : tensor<3xcomplex<f32>>
      stablehlo.return %sum, %index : tensor<3xcomplex<f32>>, tensor<i32>
    }) : (tensor<i32>) -> (tensor<3xcomplex<f32>>, tensor<i32>)
    stablehlo.return %pick#0, %pick#1 : tensor<3xcomplex<f32>>, tensor<i32>
  }, {
    stablehlo.return %w, %index : tensor<3xcomplex<f32>>, tensor<i32>
  }) : (tensor<i1>) -> (tensor<3xcomplex<f32>>, tensor<i32>)
  %real = stablehlo.real %out#0 : (tensor<3xcomplex<f32>>) -> tensor<3xf32>
  return %real, %out#1 : tensor<3xf32>, tensor<i32>
}
