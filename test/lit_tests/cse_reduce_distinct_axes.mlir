// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=cse_reduce_transposed" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// A repeated elementwise product and an outer product have different sums.
// Their dimension map [0, 0] is not a permutation, even with equal axis lengths.
func.func @different_axes(%x: tensor<3xf32>, %y: tensor<3xf32>) -> (tensor<f32>, tensor<f32>) {
  %z = stablehlo.constant dense<0.0> : tensor<f32>
  %x0 = stablehlo.broadcast_in_dim %x, dims = [0] : (tensor<3xf32>) -> tensor<3x3xf32>
  %y0 = stablehlo.broadcast_in_dim %y, dims = [0] : (tensor<3xf32>) -> tensor<3x3xf32>
  %m0 = stablehlo.multiply %x0, %y0 : tensor<3x3xf32>
  %a = stablehlo.reduce(%m0 init: %z) applies stablehlo.add across dimensions = [0, 1] : (tensor<3x3xf32>, tensor<f32>) -> tensor<f32>
  %y1 = stablehlo.broadcast_in_dim %y, dims = [1] : (tensor<3xf32>) -> tensor<3x3xf32>
  %m1 = stablehlo.multiply %x0, %y1 : tensor<3x3xf32>
  %b = stablehlo.reduce(%m1 init: %z) applies stablehlo.add across dimensions = [0, 1] : (tensor<3x3xf32>, tensor<f32>) -> tensor<f32>
  return %a, %b : tensor<f32>, tensor<f32>
}

// CHECK-LABEL: func.func @different_axes(
// CHECK: %[[A:.*]] = stablehlo.reduce
// CHECK: %[[B:.*]] = stablehlo.reduce
// CHECK: return %[[A]], %[[B]]
