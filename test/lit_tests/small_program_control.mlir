// RUN: enzymexlamlir-opt %s --lower-small-program-control | FileCheck %s

func.func @choose(%index: tensor<i32>, %a: tensor<i32>, %b: tensor<i32>, %fallback: tensor<i32>) -> tensor<i32> {
  %result = "stablehlo.case"(%index) ({
    stablehlo.return %a : tensor<i32>
  }, {
    stablehlo.return %b : tensor<i32>
  }, {
    stablehlo.return %fallback : tensor<i32>
  }) : (tensor<i32>) -> tensor<i32>
  return %result : tensor<i32>
}

// CHECK-LABEL: func.func @choose
// CHECK-DAG: %[[SELECTOR:.*]] = tensor.extract %arg0[]
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: arith.cmpi eq, %[[SELECTOR]], %[[ZERO]]
// CHECK: scf.if
// CHECK: scf.yield %arg1
// CHECK: arith.cmpi eq, %[[SELECTOR]], %[[ONE]]
// CHECK: scf.if
// CHECK: scf.yield %arg2
// CHECK: scf.yield %arg3

func.func @scatter_column(%input: tensor<2x3xi32>, %index: tensor<1xi32>, %updates: tensor<2xi32>) -> tensor<2x3xi32> {
  %result = "stablehlo.scatter"(%input, %index, %updates) <{
    scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 0>,
    indices_are_sorted = true, unique_indices = true
  }> ({
  ^bb0(%old: tensor<i32>, %new: tensor<i32>):
    %sum = stablehlo.add %old, %new : tensor<i32>
    stablehlo.return %sum : tensor<i32>
  }) : (tensor<2x3xi32>, tensor<1xi32>, tensor<2xi32>) -> tensor<2x3xi32>
  return %result : tensor<2x3xi32>
}

// CHECK-LABEL: func.func @scatter_column
// CHECK: %[[COLUMN:.*]] = arith.index_cast
// CHECK: arith.cmpi ult, %[[COLUMN]],
// CHECK: scf.if
// CHECK: scf.for
// CHECK: tensor.extract {{.*}}[{{.*}}, %[[COLUMN]]]
// CHECK: tensor.insert {{.*}}[{{.*}}, %[[COLUMN]]]
// CHECK: } else {
// CHECK-NEXT: scf.yield %arg0

func.func @scalar_check(%predicate: tensor<i1>) -> tensor<i1> {
  %true = stablehlo.constant dense<true> : tensor<i1>
  %result = stablehlo.reduce(%predicate init: %true) applies stablehlo.and across dimensions = [] : (tensor<i1>, tensor<i1>) -> tensor<i1>
  return %result : tensor<i1>
}

// CHECK-LABEL: func.func @scalar_check(
// CHECK-SAME: %[[PRED:.*]]: tensor<i1>)
// CHECK: return %[[PRED]] : tensor<i1>
