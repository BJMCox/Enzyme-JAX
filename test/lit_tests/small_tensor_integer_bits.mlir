// RUN: enzymexlamlir-opt %s --stablehlo-legalize-to-linalg --lower-small-tensor-values | FileCheck %s

func.func @main() -> (tensor<2xui32>, tensor<2xi32>) {
  %unsigned = stablehlo.constant dense<[4294967295, 2147483648]> : tensor<2xui32>
  %signed = stablehlo.constant dense<[-1, -2147483648]> : tensor<2xi32>
  return %unsigned, %signed : tensor<2xui32>, tensor<2xi32>
}

// CHECK-LABEL: func.func @main() -> (vector<2xi32>, vector<2xi32>)
// CHECK: arith.constant dense<[-1, -2147483648]> : vector<2xi32>
// CHECK: return
