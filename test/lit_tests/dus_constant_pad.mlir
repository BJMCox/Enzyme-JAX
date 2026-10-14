// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=dus_to_dynamic_pad" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// Constant indices must produce static padding without a constant-folding pass.
// CHECK-LABEL: func.func @signed_bounds
// CHECK: stablehlo.pad {{.*}}, low = [0, 3], high = [2, 0], interior = [0, 0]
// CHECK-NOT: stablehlo.dynamic_pad
func.func @signed_bounds(%update: tensor<2x2xf32>, %fill: tensor<f32>) -> tensor<4x5xf32> {
  %base = stablehlo.broadcast_in_dim %fill, dims = [] : (tensor<f32>) -> tensor<4x5xf32>
  %negative = stablehlo.constant dense<-2> : tensor<i64>
  %large = stablehlo.constant dense<9> : tensor<i64>
  %result = stablehlo.dynamic_update_slice %base, %update, %negative, %large : (tensor<4x5xf32>, tensor<2x2xf32>, tensor<i64>, tensor<i64>) -> tensor<4x5xf32>
  return %result : tensor<4x5xf32>
}

// An unsigned high bit is a large positive index, not a negative index.
// CHECK-LABEL: func.func @unsigned_bounds
// CHECK: stablehlo.pad {{.*}}, low = [2, 1], high = [0, 2], interior = [0, 0]
// CHECK-NOT: stablehlo.dynamic_pad
func.func @unsigned_bounds(%update: tensor<2x2xf32>, %fill: tensor<f32>) -> tensor<4x5xf32> {
  %base = stablehlo.broadcast_in_dim %fill, dims = [] : (tensor<f32>) -> tensor<4x5xf32>
  %large = stablehlo.constant dense<18446744073709551615> : tensor<ui64>
  %one = stablehlo.constant dense<1> : tensor<ui64>
  %result = stablehlo.dynamic_update_slice %base, %update, %large, %one : (tensor<4x5xf32>, tensor<2x2xf32>, tensor<ui64>, tensor<ui64>) -> tensor<4x5xf32>
  return %result : tensor<4x5xf32>
}
