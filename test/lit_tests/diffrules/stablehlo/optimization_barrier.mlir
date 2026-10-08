// RUN: enzymexlamlir-opt %s --enzyme | FileCheck %s

func.func private @identity(%x: tensor<3xf64>) -> (tensor<3xf64>, tensor<3xf64>) {
  %result:2 = stablehlo.optimization_barrier %x, %x : tensor<3xf64>, tensor<3xf64>
  return %result#0, %result#1 : tensor<3xf64>, tensor<3xf64>
}

func.func @main(%x: tensor<3xf64>, %dx: tensor<2x3xf64>) -> (tensor<2x3xf64>, tensor<2x3xf64>) {
  %result:2 = enzyme.fwddiff @identity(%x, %dx) <{
    activity = [#enzyme.activity<enzyme_dup>],
    ret_activity = [#enzyme.activity<enzyme_dupnoneed>, #enzyme.activity<enzyme_dupnoneed>],
    width = 2 : i64
  }> : (tensor<3xf64>, tensor<2x3xf64>) -> (tensor<2x3xf64>, tensor<2x3xf64>)
  return %result#0, %result#1 : tensor<2x3xf64>, tensor<2x3xf64>
}

// CHECK-LABEL: func.func private @fwddiffe2identity(
// CHECK-SAME: %{{.*}}: tensor<3xf64>, %[[DX:.*]]: tensor<2x3xf64>)
// CHECK: %[[D:.*]]:2 = stablehlo.optimization_barrier %[[DX]], %[[DX]] : tensor<2x3xf64>, tensor<2x3xf64>
// CHECK: return %[[D]]#0, %[[D]]#1 : tensor<2x3xf64>, tensor<2x3xf64>
