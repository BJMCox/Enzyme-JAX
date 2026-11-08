// RUN: enzymexlamlir-opt %s --lower-chlo-gamma-f32 | FileCheck %s
// RUN: enzymexlamlir-opt %s --lower-chlo-gamma-f32 --chlo-legalize-to-stablehlo | stablehlo-translate - --interpret --allow-unregistered-dialect
// The CHLO legalization misses the same bound near the roots at 1 and 2.
// A failed check aborts the interpreter, hence --crash.
// RUN: enzymexlamlir-opt %s --chlo-legalize-to-stablehlo | not --crash stablehlo-translate - --interpret --allow-unregistered-dialect

// CHECK-LABEL: func.func @lgamma_f32
// CHECK-NOT: chlo.lgamma
// CHECK: stablehlo.gather
func.func @lgamma_f32(%x: tensor<17xf32>) -> tensor<17xf32> {
  %y = chlo.lgamma %x : tensor<17xf32> -> tensor<17xf32>
  return %y : tensor<17xf32>
}

// CHECK-LABEL: func.func @lgamma_f64
// CHECK: chlo.lgamma
func.func @lgamma_f64(%x: tensor<f64>) -> tensor<f64> {
  %y = chlo.lgamma %x : tensor<f64> -> tensor<f64>
  return %y : tensor<f64>
}

// CHECK-LABEL: func.func @lgamma_specials
func.func @lgamma_specials(%x: tensor<5xf32>) -> tensor<5xf32> {
  %y = chlo.lgamma %x : tensor<5xf32> -> tensor<5xf32>
  return %y : tensor<5xf32>
}

func.func @main() {
  // Points cover each branch: reflection, root windows, the two polynomial
  // ranges, the Stirling series. References are mpmath rounded to float32.
  %x = stablehlo.constant dense<[1.0e-03, 0.3, 0.999, 1.001, 1.25, 1.4616321, 1.9995, 2.003, 2.4, 3.7, 4.5, 37.25, 1.0e+05, -0.5, -2.457, -3.3, -7.9999]> : tensor<17xf32>
  %ref = stablehlo.constant dense<[6.907179, 1.0957979, 5.7803106e-04, -5.764205e-04, -0.09827184, -0.12148629, -2.1129624e-04, 1.2712623e-03, 0.21685939, 1.4280723, 2.4537365, 96.61989, 1.0512878e+06, 1.2655121, 3.7470283e-05, -0.82435566, -1.3954053]> : tensor<17xf32>
  %y = func.call @lgamma_f32(%x) : (tensor<17xf32>) -> tensor<17xf32>
  check.expect_close %y, %ref, max_ulp_difference = 2 : tensor<17xf32>, tensor<17xf32>

  // Poles give +inf; lgamma(1) and lgamma(2) give +0.
  %s = stablehlo.constant dense<[0.0, -0.0, -3.0, 1.0, 2.0]> : tensor<5xf32>
  %sy = func.call @lgamma_specials(%s) : (tensor<5xf32>) -> tensor<5xf32>
  check.expect_eq_const %sy, dense<[0x7F800000, 0x7F800000, 0x7F800000, 0.0, 0.0]> : tensor<5xf32>
  return
}
