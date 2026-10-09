// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td='patterns=mul_reduce_slice_fusion' --transform-interpreter --enzyme-hlo-remove-transform --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @floating_pair
// CHECK-NOT: stablehlo.reduce
// CHECK: stablehlo.multiply
// CHECK-NOT: stablehlo.reduce
// CHECK: return
func.func @floating_pair(%x: tensor<2xf32>) -> tensor<1xf32> {
  %a = stablehlo.slice %x [0:1] : (tensor<2xf32>) -> tensor<1xf32>
  %b = stablehlo.slice %x [1:2] : (tensor<2xf32>) -> tensor<1xf32>
  %p = stablehlo.multiply %a, %b : tensor<1xf32>
  return %p : tensor<1xf32>
}

// CHECK-LABEL: func.func @integer_chain
// CHECK: stablehlo.reduce
// CHECK: return
func.func @integer_chain(%x: tensor<4xi32>) -> tensor<1xi32> {
  %a = stablehlo.slice %x [0:1] : (tensor<4xi32>) -> tensor<1xi32>
  %b = stablehlo.slice %x [1:2] : (tensor<4xi32>) -> tensor<1xi32>
  %c = stablehlo.slice %x [2:3] : (tensor<4xi32>) -> tensor<1xi32>
  %d = stablehlo.slice %x [3:4] : (tensor<4xi32>) -> tensor<1xi32>
  %ab = stablehlo.multiply %a, %b : tensor<1xi32>
  %abc = stablehlo.multiply %ab, %c : tensor<1xi32>
  %p = stablehlo.multiply %abc, %d : tensor<1xi32>
  return %p : tensor<1xi32>
}
