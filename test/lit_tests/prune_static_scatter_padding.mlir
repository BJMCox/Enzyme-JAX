// RUN: enzymexlamlir-opt %s --transform-interpreter | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %functions = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %functions {
      transform.apply_patterns.enzyme_hlo.prune_static_scatter_padding
    } : !transform.any_op
    transform.yield
  }

  func.func @main(%updates: tensor<2x2x3x2xf32>) -> tensor<2x4x2xf32> {
    %zero = stablehlo.constant dense<0.0> : tensor<2x6x2xf32>
    %indices = stablehlo.constant dense<[[[[2], [0], [5]], [[1], [3], [5]]], [[[2], [0], [5]], [[1], [3], [5]]]]> : tensor<2x2x3x1xi32>
    %scattered = "stablehlo.scatter"(%zero, %indices, %updates) ({
      ^bb0(%old: tensor<f32>, %update: tensor<f32>):
        %sum = stablehlo.add %old, %update : tensor<f32>
        stablehlo.return %sum : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [3], inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [0], scatter_dims_to_operand_dims = [1], index_vector_dim = 3>, indices_are_sorted = false, unique_indices = false}
        : (tensor<2x6x2xf32>, tensor<2x2x3x1xi32>, tensor<2x2x3x2xf32>) -> tensor<2x6x2xf32>
    %result = stablehlo.slice %scattered [0:2, 0:4, 0:2] : (tensor<2x6x2xf32>) -> tensor<2x4x2xf32>
    return %result : tensor<2x4x2xf32>
  }

  func.func @promoted(%updates: tensor<2x3x2xf32>) -> tensor<4x2xf64> {
    %zero = stablehlo.constant dense<0.0> : tensor<6x2xf32>
    %indices = stablehlo.constant dense<[[[2], [0], [5]], [[1], [3], [5]]]> : tensor<2x3x1xi32>
    %scattered = "stablehlo.scatter"(%zero, %indices, %updates) ({
      ^bb0(%old: tensor<f64>, %update: tensor<f64>):
        %sum = stablehlo.add %old, %update : tensor<f64>
        stablehlo.return %sum : tensor<f64>
    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [2], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 2>, indices_are_sorted = false, unique_indices = false}
        : (tensor<6x2xf32>, tensor<2x3x1xi32>, tensor<2x3x2xf32>) -> tensor<6x2xf64>
    %result = stablehlo.slice %scattered [0:4, 0:2] : (tensor<6x2xf64>) -> tensor<4x2xf64>
    return %result : tensor<4x2xf64>
  }
}

// CHECK-LABEL: func.func {{(private )?}}@main(
// CHECK: stablehlo.gather
// CHECK: stablehlo.optimization_barrier
// CHECK: stablehlo.add
// CHECK-NOT: stablehlo.scatter
// CHECK: return {{.*}} : tensor<2x4x2xf32>

// CHECK-LABEL: func.func {{(private )?}}@promoted(
// CHECK: stablehlo.scatter
// CHECK: -> tensor<6x2xf64>
// CHECK: stablehlo.slice
