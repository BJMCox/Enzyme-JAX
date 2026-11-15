// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_dup argTys=enzyme_dup,enzyme_dup mode=ForwardMode" | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=main outfn= retTys=enzyme_active argTys=enzyme_active,enzyme_dup mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s --check-prefix=REVERSE

module {
  func.func public @main(%arg3: tensor<3xf32>, %arg6: tensor<5xf32>) -> (tensor<5xf32>) {
    %c_0 = stablehlo.constant dense<[[3], [2], [1]]> : tensor<3x1xi32>
    %33 = "stablehlo.scatter"(%arg6, %c_0, %arg3) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
    ^bb0(%arg9: tensor<f32>, %arg10: tensor<f32>):
      stablehlo.return %arg10 : tensor<f32>
    }) : (tensor<5xf32>, tensor<3x1xi32>, tensor<3xf32>) -> tensor<5xf32>
    return %33 : tensor<5xf32>
  }
}

// FORWARD:  func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>, %arg2: tensor<5xf32>, %arg3: tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>) {
// FORWARD-NEXT:    %c = stablehlo.constant dense<{{\[\[}}3], [2], [1{{\]\]}}> : tensor<3x1xi32>
// FORWARD-NEXT:    %0:2 = "stablehlo.scatter"(%arg2, %arg3, %c, %arg0, %arg1) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false}> ({
// FORWARD-NEXT:    ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>):
// FORWARD-NEXT:      stablehlo.return %arg6, %arg7 : tensor<f32>, tensor<f32>
// FORWARD-NEXT:    }) : (tensor<5xf32>, tensor<5xf32>, tensor<3x1xi32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<5xf32>, tensor<5xf32>)
// FORWARD-NEXT:    return %0#0, %0#1 : tensor<5xf32>, tensor<5xf32>
// FORWARD-NEXT:  }

// REVERSE: func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<5xf32>, %arg2: tensor<5xf32>, %arg3: tensor<5xf32>) -> tensor<3xf32> {
// REVERSE-DAG: [[ZERO_UPDATES:%.*]] = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
// REVERSE-DAG: [[ZERO_SEED:%.*]] = stablehlo.constant dense<0.000000e+00> : tensor<5xf32>
// REVERSE-DAG: [[ONE:%.*]] = stablehlo.constant dense<1> : tensor<5xi64>
// REVERSE-DAG: [[ONE_UPDATE:%.*]] = stablehlo.constant dense<1> : tensor<3xi64>
// REVERSE-DAG: [[ZERO_IDS:%.*]] = stablehlo.constant dense<0> : tensor<5xi64>
// REVERSE-DAG: [[INDICES:%.*]] = stablehlo.constant dense<{{\[\[}}3], [2], [1{{\]\]}}> : tensor<3x1xi32>
// REVERSE: [[IOTA:%.*]] = stablehlo.iota dim = 0 : tensor<3xi64>
// REVERSE: [[SHAPED_IDS:%.*]] = stablehlo.reshape [[IOTA]] : (tensor<3xi64>) -> tensor<3xi64>
// REVERSE: [[IDS:%.*]] = stablehlo.add [[SHAPED_IDS]], [[ONE_UPDATE]] : tensor<3xi64>
// REVERSE: [[WINNER:%.*]] = "stablehlo.scatter"([[ZERO_IDS]], [[INDICES]], [[IDS]]) {{.*}}unique_indices = false
// REVERSE: ^bb0([[OLD_ID:%.*]]: tensor<i64>, [[NEW_ID:%.*]]: tensor<i64>):
// REVERSE-NEXT: [[MAX:%.*]] = stablehlo.maximum [[OLD_ID]], [[NEW_ID]] : tensor<i64>
// REVERSE-NEXT: stablehlo.return [[MAX]] : tensor<i64>
// REVERSE: [[WRITTEN:%.*]] = stablehlo.compare NE, [[WINNER]], [[ZERO_IDS]]
// REVERSE: [[INDEX:%.*]] = stablehlo.subtract [[WINNER]], [[ONE]] : tensor<5xi64>
// REVERSE: [[FLAT_INDEX:%.*]] = stablehlo.reshape [[INDEX]] : (tensor<5xi64>) -> tensor<5x1xi64>
// REVERSE: [[SEED:%.*]] = stablehlo.select [[WRITTEN]], %arg3, [[ZERO_SEED]]
// REVERSE: [[FLAT_SEED:%.*]] = stablehlo.reshape [[SEED]] : (tensor<5xf32>) -> tensor<5xf32>
// REVERSE: [[UPDATE:%.*]] = "stablehlo.scatter"([[ZERO_UPDATES]], [[FLAT_INDEX]], [[FLAT_SEED]]) {{.*}}scatter_dims_to_operand_dims = [0], index_vector_dim = 1>{{.*}}unique_indices = false
// REVERSE: ^bb0([[OLD_SEED:%.*]]: tensor<f32>, [[NEW_SEED:%.*]]: tensor<f32>):
// REVERSE-NEXT: [[SUM:%.*]] = stablehlo.add [[OLD_SEED]], [[NEW_SEED]] : tensor<f32>
// REVERSE-NEXT: stablehlo.return [[SUM]] : tensor<f32>
// REVERSE: [[RESULT:%.*]] = stablehlo.reshape [[UPDATE]] : (tensor<3xf32>) -> tensor<3xf32>
// REVERSE-NEXT: return [[RESULT]] : tensor<3xf32>
