// RUN: enzymexlamlir-opt %s --transform-interpreter | FileCheck %s

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    %functions = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %functions {
      transform.apply_patterns.enzyme_hlo.compact_static_scatter
    } : !transform.any_op
    transform.yield
  }

  func.func @main(%base: tensor<2147483654xf32>, %updates: tensor<2xf32>) -> tensor<2147483654xf32> {
    %indices = stablehlo.constant dense<[[2147483650], [2147483648]]> : tensor<2x1xi64>
    %result = "stablehlo.scatter"(%base, %indices, %updates) ({
      ^bb0(%old: tensor<f32>, %update: tensor<f32>):
        %sum = stablehlo.add %old, %update : tensor<f32>
        stablehlo.return %sum : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, indices_are_sorted = false, unique_indices = true}
        : (tensor<2147483654xf32>, tensor<2x1xi64>, tensor<2xf32>) -> tensor<2147483654xf32>
    return %result : tensor<2147483654xf32>
  }

  func.func @i32_start(%base: tensor<2147483654xf32>, %updates: tensor<2xf32>) -> tensor<2147483654xf32> {
    %indices = stablehlo.constant dense<[[2147483649], [2147483647]]> : tensor<2x1xi64>
    %result = "stablehlo.scatter"(%base, %indices, %updates) ({
      ^bb0(%old: tensor<f32>, %update: tensor<f32>):
        stablehlo.return %update : tensor<f32>
    }) {scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, indices_are_sorted = false, unique_indices = true}
        : (tensor<2147483654xf32>, tensor<2x1xi64>, tensor<2xf32>) -> tensor<2147483654xf32>
    return %result : tensor<2147483654xf32>
  }
}

// CHECK-LABEL: func.func {{(private )?}}@main(
// CHECK-DAG: stablehlo.optimization_barrier
// CHECK-DAG: %[[START64:.*]] = stablehlo.constant dense<2147483648> : tensor<i64>
// CHECK: stablehlo.dynamic_update_slice {{.*}}, %[[START64]] : (tensor<2147483654xf32>, tensor<3xf32>, tensor<i64>)

// CHECK-LABEL: func.func {{(private )?}}@i32_start(
// CHECK: %[[START32:.*]] = stablehlo.constant dense<2147483647> : tensor<i32>
// CHECK: stablehlo.dynamic_update_slice {{.*}}, %[[START32]] : (tensor<2147483654xf32>, tensor<3xf32>, tensor<i32>)
