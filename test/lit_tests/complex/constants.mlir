// RUN: enzymexlamlir-opt --lower-complex-operations="concat-dimension=first" %s | FileCheck %s --check-prefixes=CHECK,FIRST --implicit-check-not=unrealized_conversion_cast --implicit-check-not='complex<'
// RUN: enzymexlamlir-opt --lower-complex-operations="concat-dimension=last" %s | FileCheck %s --check-prefixes=CHECK,LAST --implicit-check-not=unrealized_conversion_cast --implicit-check-not='complex<'

// CHECK-LABEL: @dense(
// CHECK: %[[DATA:.*]] = stablehlo.constant dense<{{.*}}> : tensor<3x2xf32>
// FIRST: stablehlo.transpose %[[DATA]], dims = [1, 0]
func.func @dense() -> tensor<3xf32> {
  %z = stablehlo.constant dense<[(1.0, -2.0), (3.0, 4.0), (-5.0, 6.0)]> : tensor<3xcomplex<f32>>
  %i = stablehlo.imag %z : (tensor<3xcomplex<f32>>) -> tensor<3xf32>
  return %i : tensor<3xf32>
}

// CHECK-LABEL: @splat(
// CHECK: %[[PAIR:.*]] = stablehlo.constant dense<[1.000000e+00, -2.000000e+00]> : tensor<2xf64>
// FIRST: stablehlo.broadcast_in_dim %[[PAIR]], dims = [0] : (tensor<2xf64>) -> tensor<2x100000xf64>
// LAST:  stablehlo.broadcast_in_dim %[[PAIR]], dims = [1] : (tensor<2xf64>) -> tensor<100000x2xf64>
func.func @splat() -> tensor<100000xf64> {
  %z = stablehlo.constant dense<(1.0, -2.0)> : tensor<100000xcomplex<f64>>
  %i = stablehlo.imag %z : (tensor<100000xcomplex<f64>>) -> tensor<100000xf64>
  return %i : tensor<100000xf64>
}

// CHECK-LABEL: @resource(
// CHECK: %[[PAIR:.*]] = stablehlo.constant dense_resource<values> : tensor<2xf32>
// CHECK: stablehlo.slice %[[PAIR]] [1:2]
func.func @resource() -> tensor<f32> {
  %z = stablehlo.constant dense_resource<values> : tensor<complex<f32>>
  %i = stablehlo.imag %z : (tensor<complex<f32>>) -> tensor<f32>
  return %i : tensor<f32>
}

{-#
  dialect_resources: {
    builtin: {
      values: "0x040000000000803F000000C0"
    }
  }
#-}
