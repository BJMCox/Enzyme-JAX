// RUN: enzymexlamlir-opt %s --lower-chlo-gamma-f32 | FileCheck %s
// RUN: enzymexlamlir-opt %s --lower-chlo-gamma-f32 --chlo-legalize-to-stablehlo | stablehlo-translate - --interpret --allow-unregistered-dialect
// The CHLO legalization misses the same bound near the root at 1.4616.
// A failed check aborts the interpreter, hence --crash.
// RUN: enzymexlamlir-opt %s --chlo-legalize-to-stablehlo | not --crash stablehlo-translate - --interpret --allow-unregistered-dialect

// CHECK-LABEL: func.func @digamma_f32
// CHECK-NOT: chlo.digamma
// CHECK: stablehlo.log
func.func @digamma_f32(%x: tensor<11xf32>) -> tensor<11xf32> {
  %y = chlo.digamma %x : tensor<11xf32> -> tensor<11xf32>
  return %y : tensor<11xf32>
}

// CHECK-LABEL: func.func @digamma_f64
// CHECK: chlo.digamma
func.func @digamma_f64(%x: tensor<f64>) -> tensor<f64> {
  %y = chlo.digamma %x : tensor<f64> -> tensor<f64>
  return %y : tensor<f64>
}

// CHECK-LABEL: func.func @digamma_specials
func.func @digamma_specials(%x: tensor<3xf32>) -> tensor<3xf32> {
  %y = chlo.digamma %x : tensor<3xf32> -> tensor<3xf32>
  return %y : tensor<3xf32>
}

func.func @main() {
  // Points cover reflection, the recurrence, the root at 1.4616 and the
  // asymptotic series. References are mpmath rounded to float32.
  %x = stablehlo.constant dense<[1.0e-03, 0.75, 1.45, 1.4616321, 1.47, 2.2, 3.3, 5.9, 7.5, 150.0, -0.3]> : tensor<11xf32>
  %ref = stablehlo.constant dense<[-1000.5755, -1.0858608, -0.011316376, -1.2036052e-08, 0.008066516, 0.54429346, 1.0348225, 1.6878195, 1.9467574, 5.0072985, 2.1133096]> : tensor<11xf32>
  %y = func.call @digamma_f32(%x) : (tensor<11xf32>) -> tensor<11xf32>
  check.expect_close %y, %ref, max_ulp_difference = 2 : tensor<11xf32>, tensor<11xf32>

  %s = stablehlo.constant dense<[0.0, -0.0, 0x7F800000]> : tensor<3xf32>
  %sy = func.call @digamma_specials(%s) : (tensor<3xf32>) -> tensor<3xf32>
  check.expect_eq_const %sy, dense<[0xFF800000, 0xFF800000, 0x7F800000]> : tensor<3xf32>
  return
}
