// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(enzyme-hlo-generate-td{patterns=scatter_op_canon<16>},transform-interpreter,enzyme-hlo-remove-transform)' | FileCheck %s

// An empty index grid has no offset from which to infer an affine grid.
// CHECK-LABEL: func.func @empty_index_grid
// CHECK: "stablehlo.scatter"
// CHECK: return
func.func @empty_index_grid(%a: tensor<4x4xf32>, %u: tensor<0x4xf32>) -> tensor<4x4xf32> {
  %idx = stablehlo.constant dense<> : tensor<0x1xi32>
  %r = "stablehlo.scatter"(%a, %idx, %u) <{indices_are_sorted = true, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = true}> ({
  ^bb0(%old: tensor<f32>, %new: tensor<f32>):
    stablehlo.return %new : tensor<f32>
  }) : (tensor<4x4xf32>, tensor<0x1xi32>, tensor<0x4xf32>) -> tensor<4x4xf32>
  return %r : tensor<4x4xf32>
}
