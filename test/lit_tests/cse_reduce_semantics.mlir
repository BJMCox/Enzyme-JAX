// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=cse_reduce_transposed" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Opposite comparisons of the same inputs are different computations.
func.func @different_predicates(%x: tensor<3xf32>, %y: tensor<3xf32>) -> (tensor<i1>, tensor<i1>) {
  %z = stablehlo.constant dense<false> : tensor<i1>
  %lt = stablehlo.compare LT, %x, %y, FLOAT : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
  %a = stablehlo.reduce(%lt init: %z) applies stablehlo.or across dimensions = [0] : (tensor<3xi1>, tensor<i1>) -> tensor<i1>
  %gt = stablehlo.compare GT, %x, %y, FLOAT : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>
  %b = stablehlo.reduce(%gt init: %z) applies stablehlo.or across dimensions = [0] : (tensor<3xi1>, tensor<i1>) -> tensor<i1>
  return %a, %b : tensor<i1>, tensor<i1>
}

// CHECK-LABEL: func.func @different_predicates(
// CHECK: %[[LT:.*]] = stablehlo.compare LT,
// CHECK: %[[A:.*]] = stablehlo.reduce(%[[LT]]
// CHECK: %[[GT:.*]] = stablehlo.compare GT,
// CHECK: %[[B:.*]] = stablehlo.reduce(%[[GT]]
// CHECK: return %[[A]], %[[B]]

// Equal final types do not make different intermediate rounding equivalent.
func.func @different_precision(%x: tensor<3xf32>) -> (tensor<f32>, tensor<f32>) {
  %z = stablehlo.constant dense<0.0> : tensor<f32>
  %half = stablehlo.convert %x : (tensor<3xf32>) -> tensor<3xf16>
  %rounded = stablehlo.convert %half : (tensor<3xf16>) -> tensor<3xf32>
  %a = stablehlo.reduce(%rounded init: %z) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
  %double = stablehlo.convert %x : (tensor<3xf32>) -> tensor<3xf64>
  %exact = stablehlo.convert %double : (tensor<3xf64>) -> tensor<3xf32>
  %b = stablehlo.reduce(%exact init: %z) applies stablehlo.add across dimensions = [0] : (tensor<3xf32>, tensor<f32>) -> tensor<f32>
  return %a, %b : tensor<f32>, tensor<f32>
}

// CHECK-LABEL: func.func @different_precision(
// CHECK: %[[A:.*]] = stablehlo.reduce
// CHECK: %[[B:.*]] = stablehlo.reduce
// CHECK: return %[[A]], %[[B]]
