// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=dus_to_i32" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Unsigned high bits must not become negative and reverse index clamping.
// CHECK-LABEL: func.func @unsigned_max(
// CHECK: %[[INDEX:.*]] = stablehlo.constant dense<18446744073709551615> : tensor<ui64>
// CHECK: %[[OUT:.*]] = stablehlo.dynamic_update_slice {{.*}}, %[[INDEX]] : (tensor<2xf32>, tensor<1xf32>, tensor<ui64>)
// CHECK-NEXT: return %[[OUT]] : tensor<2xf32>
func.func @unsigned_max(%x: tensor<2xf32>, %u: tensor<1xf32>) -> tensor<2xf32> {
  %index = stablehlo.constant dense<18446744073709551615> : tensor<ui64>
  %out = stablehlo.dynamic_update_slice %x, %u, %index : (tensor<2xf32>, tensor<1xf32>, tensor<ui64>) -> tensor<2xf32>
  return %out : tensor<2xf32>
}

// The signed i32 boundary also applies to unsigned 32-bit inputs.
// CHECK-LABEL: func.func @unsigned_out_of_range(
// CHECK: %[[INDEX:.*]] = stablehlo.constant dense<2147483648> : tensor<ui32>
// CHECK: %[[OUT:.*]] = stablehlo.dynamic_update_slice {{.*}}, %[[INDEX]] : (tensor<2xf32>, tensor<1xf32>, tensor<ui32>)
// CHECK-NEXT: return %[[OUT]] : tensor<2xf32>
func.func @unsigned_out_of_range(%x: tensor<2xf32>, %u: tensor<1xf32>) -> tensor<2xf32> {
  %index = stablehlo.constant dense<2147483648> : tensor<ui32>
  %out = stablehlo.dynamic_update_slice %x, %u, %index : (tensor<2xf32>, tensor<1xf32>, tensor<ui32>) -> tensor<2xf32>
  return %out : tensor<2xf32>
}

// An unsigned source sign bit must survive widening as a positive value.
// CHECK-LABEL: func.func @unsigned_in_range(
// CHECK: %[[INDEX:.*]] = stablehlo.constant dense<65535> : tensor<i32>
// CHECK: %[[OUT:.*]] = stablehlo.dynamic_update_slice {{.*}}, %[[INDEX]] : (tensor<2xf32>, tensor<1xf32>, tensor<i32>)
// CHECK-NEXT: return %[[OUT]] : tensor<2xf32>
func.func @unsigned_in_range(%x: tensor<2xf32>, %u: tensor<1xf32>) -> tensor<2xf32> {
  %index = stablehlo.constant dense<65535> : tensor<ui16>
  %out = stablehlo.dynamic_update_slice %x, %u, %index : (tensor<2xf32>, tensor<1xf32>, tensor<ui16>) -> tensor<2xf32>
  return %out : tensor<2xf32>
}

// Extending a narrow signed index must preserve its negative value.
// CHECK-LABEL: func.func @signed_negative(
// CHECK: %[[INDEX:.*]] = stablehlo.constant dense<-1> : tensor<i32>
// CHECK: %[[OUT:.*]] = stablehlo.dynamic_update_slice {{.*}}, %[[INDEX]] : (tensor<2xf32>, tensor<1xf32>, tensor<i32>)
// CHECK-NEXT: return %[[OUT]] : tensor<2xf32>
func.func @signed_negative(%x: tensor<2xf32>, %u: tensor<1xf32>) -> tensor<2xf32> {
  %index = stablehlo.constant dense<-1> : tensor<i8>
  %out = stablehlo.dynamic_update_slice %x, %u, %index : (tensor<2xf32>, tensor<1xf32>, tensor<i8>) -> tensor<2xf32>
  return %out : tensor<2xf32>
}
