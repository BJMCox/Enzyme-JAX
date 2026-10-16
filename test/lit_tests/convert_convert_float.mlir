// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=convert_convert_float" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// Widening preserves the value supplied to the final conversion.
// CHECK-LABEL: func.func @widen_roundtrip(
// CHECK-SAME: %[[X:.*]]: tensor<f32>
// CHECK-NOT: stablehlo.convert
// CHECK: return %[[X]] : tensor<f32>
func.func @widen_roundtrip(%x: tensor<f32>) -> tensor<f32> {
  %wide = stablehlo.convert %x : (tensor<f32>) -> tensor<f64>
  %out = stablehlo.convert %wide : (tensor<f64>) -> tensor<f32>
  return %out : tensor<f32>
}

// CHECK-LABEL: func.func @widen_chain(
// CHECK-SAME: %[[X:.*]]: tensor<f16>
// CHECK: %[[OUT:.*]] = stablehlo.convert %[[X]] : (tensor<f16>) -> tensor<f64>
// CHECK-NEXT: return %[[OUT]] : tensor<f64>
func.func @widen_chain(%x: tensor<f16>) -> tensor<f64> {
  %wide = stablehlo.convert %x : (tensor<f16>) -> tensor<f32>
  %out = stablehlo.convert %wide : (tensor<f32>) -> tensor<f64>
  return %out : tensor<f64>
}

// Equal bit widths do not imply equal ranges: 2^100 overflows f16.
// CHECK-LABEL: func.func @different_ranges(
// CHECK: %[[MID:.*]] = stablehlo.convert {{.*}} : (tensor<bf16>) -> tensor<f16>
// CHECK-NEXT: %[[OUT:.*]] = stablehlo.convert %[[MID]] : (tensor<f16>) -> tensor<f32>
// CHECK-NEXT: return %[[OUT]] : tensor<f32>
func.func @different_ranges(%x: tensor<bf16>) -> tensor<f32> {
  %half = stablehlo.convert %x : (tensor<bf16>) -> tensor<f16>
  %out = stablehlo.convert %half : (tensor<f16>) -> tensor<f32>
  return %out : tensor<f32>
}

// FNUZ loses signed zero and cannot preserve infinity.
// CHECK-LABEL: func.func @nonfinite_and_zero(
// CHECK: %[[MID:.*]] = stablehlo.convert {{.*}} : (tensor<f8E4M3>) -> tensor<f8E4M3FNUZ>
// CHECK-NEXT: %[[OUT:.*]] = stablehlo.convert %[[MID]] : (tensor<f8E4M3FNUZ>) -> tensor<f8E4M3>
// CHECK-NEXT: return %[[OUT]] : tensor<f8E4M3>
func.func @nonfinite_and_zero(%x: tensor<f8E4M3>) -> tensor<f8E4M3> {
  %unsigned_zero = stablehlo.convert %x : (tensor<f8E4M3>) -> tensor<f8E4M3FNUZ>
  %out = stablehlo.convert %unsigned_zero : (tensor<f8E4M3FNUZ>) -> tensor<f8E4M3>
  return %out : tensor<f8E4M3>
}
