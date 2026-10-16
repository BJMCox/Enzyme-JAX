// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=dus_reshape" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// The unsigned row index clamps to row 1, which starts at flat index 3.
func.func @unsigned_reshape(%x: tensor<6xf32>, %u: tensor<1x1xf32>) -> tensor<2x3xf32> {
  %large = stablehlo.constant dense<18446744073709551615> : tensor<ui64>
  %zero = stablehlo.constant dense<0> : tensor<ui64>
  %r = stablehlo.reshape %x : (tensor<6xf32>) -> tensor<2x3xf32>
  %d = stablehlo.dynamic_update_slice %r, %u, %large, %zero : (tensor<2x3xf32>, tensor<1x1xf32>, tensor<ui64>, tensor<ui64>) -> tensor<2x3xf32>
  return %d : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @unsigned_reshape(
// CHECK: %[[START:.*]] = stablehlo.constant dense<3> : tensor<ui64>
// CHECK: stablehlo.dynamic_update_slice {{.*}}, %[[START]] : (tensor<6xf32>, tensor<1xf32>, tensor<ui64>)

// The same high bits in a signed index represent -1 and clamp to row 0.
func.func @signed_reshape(%x: tensor<6xf32>, %u: tensor<1x1xf32>) -> tensor<2x3xf32> {
  %negative = stablehlo.constant dense<-1> : tensor<i64>
  %zero = stablehlo.constant dense<0> : tensor<i64>
  %r = stablehlo.reshape %x : (tensor<6xf32>) -> tensor<2x3xf32>
  %d = stablehlo.dynamic_update_slice %r, %u, %negative, %zero : (tensor<2x3xf32>, tensor<1x1xf32>, tensor<i64>, tensor<i64>) -> tensor<2x3xf32>
  return %d : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @signed_reshape(
// CHECK: %[[START:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK: stablehlo.dynamic_update_slice {{.*}}, %[[START]] : (tensor<6xf32>, tensor<1xf32>, tensor<i64>)
