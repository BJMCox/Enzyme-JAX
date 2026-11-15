// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(enzyme-hlo-generate-td{patterns=scatter_op_canon<16>},transform-interpreter,enzyme-hlo-remove-transform)' | FileCheck %s

// A single index addresses the whole two-element update window.
// CHECK-LABEL: func.func @single_index_window
// CHECK: "stablehlo.scatter"
// CHECK: return
func.func @single_index_window(%input: tensor<4xi32>, %update: tensor<2xi32>) -> tensor<4xi32> {
  %index = stablehlo.constant dense<2> : tensor<1xi32>
  %result = "stablehlo.scatter"(%input, %index, %update) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
  ^bb0(%old: tensor<i32>, %new: tensor<i32>):
    stablehlo.return %new : tensor<i32>
  }) : (tensor<4xi32>, tensor<1xi32>, tensor<2xi32>) -> tensor<4xi32>
  return %result : tensor<4xi32>
}
