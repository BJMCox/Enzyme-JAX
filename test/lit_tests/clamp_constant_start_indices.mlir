// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=clamp_constant_start_indices" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Clamp each axis before a backend combines indices into a flat offset.
// CHECK-LABEL: func.func @unsigned_update(
// CHECK: %[[ONE:.*]] = stablehlo.constant dense<1> : tensor<ui16>
// CHECK: stablehlo.dynamic_update_slice %arg0, %arg1, %[[ONE]], %arg2
func.func @unsigned_update(%x: tensor<2x3xf32>, %u: tensor<1x1xf32>, %j: tensor<ui16>) -> tensor<2x3xf32> {
  %i = stablehlo.constant dense<65535> : tensor<ui16>
  %out = stablehlo.dynamic_update_slice %x, %u, %i, %j : (tensor<2x3xf32>, tensor<1x1xf32>, tensor<ui16>, tensor<ui16>) -> tensor<2x3xf32>
  return %out : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @signed_slice(
// CHECK-DAG: %[[ZERO:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-DAG: %[[TWO:.*]] = stablehlo.constant dense<2> : tensor<i64>
// CHECK: stablehlo.dynamic_slice %arg0, %[[ZERO]], %[[TWO]], sizes = [1, 1]
func.func @signed_slice(%x: tensor<2x3xf32>) -> tensor<1x1xf32> {
  %i = stablehlo.constant dense<-1> : tensor<i64>
  %j = stablehlo.constant dense<2147483648> : tensor<i64>
  %out = stablehlo.dynamic_slice %x, %i, %j, sizes = [1, 1] : (tensor<2x3xf32>, tensor<i64>, tensor<i64>) -> tensor<1x1xf32>
  return %out : tensor<1x1xf32>
}

// Unknown bounds stay dynamic; known bounds still clamp.
// CHECK-LABEL: func.func @dynamic_update(
// CHECK-DAG: %[[OLD:.*]] = stablehlo.constant dense<65535> : tensor<ui16>
// CHECK-DAG: %[[THREE:.*]] = stablehlo.constant dense<3> : tensor<ui16>
// CHECK: stablehlo.dynamic_update_slice %arg0, %arg1, %[[OLD]], %[[OLD]], %[[THREE]]
func.func @dynamic_update(%x: tensor<?x3x4xf32>, %u: tensor<1x?x1xf32>) -> tensor<?x3x4xf32> {
  %i = stablehlo.constant dense<65535> : tensor<ui16>
  %out = stablehlo.dynamic_update_slice %x, %u, %i, %i, %i : (tensor<?x3x4xf32>, tensor<1x?x1xf32>, tensor<ui16>, tensor<ui16>, tensor<ui16>) -> tensor<?x3x4xf32>
  return %out : tensor<?x3x4xf32>
}

// Do not truncate the shape bound to the narrower index type.
// CHECK-LABEL: func.func @narrow_indices(
// CHECK-DAG: %[[OLD:.*]] = stablehlo.constant dense<250> : tensor<ui8>
// CHECK-DAG: %[[TWO:.*]] = stablehlo.constant dense<2> : tensor<ui8>
// CHECK: stablehlo.dynamic_slice %arg0, %[[OLD]], %[[TWO]], sizes = [1, 1]
func.func @narrow_indices(%x: tensor<300x3xf32>) -> tensor<1x1xf32> {
  %i = stablehlo.constant dense<250> : tensor<ui8>
  %out = stablehlo.dynamic_slice %x, %i, %i, sizes = [1, 1] : (tensor<300x3xf32>, tensor<ui8>, tensor<ui8>) -> tensor<1x1xf32>
  return %out : tensor<1x1xf32>
}

// The bound is operand extent minus update extent, including full-size axes.
// CHECK-LABEL: func.func @wide_unsigned_update(
// CHECK-DAG: %[[ZERO:.*]] = stablehlo.constant dense<0> : tensor<ui64>
// CHECK-DAG: %[[ONE:.*]] = stablehlo.constant dense<1> : tensor<ui64>
// CHECK: stablehlo.dynamic_update_slice %arg0, %arg1, %[[ZERO]], %[[ONE]]
func.func @wide_unsigned_update(%x: tensor<2x3xf32>, %u: tensor<2x2xf32>) -> tensor<2x3xf32> {
  %i = stablehlo.constant dense<18446744073709551615> : tensor<ui64>
  %out = stablehlo.dynamic_update_slice %x, %u, %i, %i : (tensor<2x3xf32>, tensor<2x2xf32>, tensor<ui64>, tensor<ui64>) -> tensor<2x3xf32>
  return %out : tensor<2x3xf32>
}
