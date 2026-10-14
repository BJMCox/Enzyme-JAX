// RUN: enzymexlamlir-opt %s --enzyme --arith-raise="stablehlo=true" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize | FileCheck %s

// The previous state is already taped for the parameter gradient. Reverse can
// read the next row or final forward state instead of recomputing it.
module {
  func.func private @loop(%a: tensor<4xf64>) -> tensor<4xf64> {
    %zero = stablehlo.constant dense<0> : tensor<i64>
    %one = stablehlo.constant dense<1> : tensor<i64>
    %eight = stablehlo.constant dense<8> : tensor<i64>
    %init = stablehlo.constant dense<1.0> : tensor<4xf64>
    %offset = stablehlo.constant dense<0.2> : tensor<4xf64>
    %total = stablehlo.constant dense<0.0> : tensor<4xf64>
    %r:3 = stablehlo.while(%i = %zero, %state = %init, %sum = %total) : tensor<i64>, tensor<4xf64>, tensor<4xf64>
    cond {
      %c = stablehlo.compare LT, %i, %eight : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %product = stablehlo.multiply %a, %state : tensor<4xf64>
      %next = stablehlo.add %offset, %product : tensor<4xf64>
      %score = stablehlo.log %next : tensor<4xf64>
      %sum_next = stablehlo.add %sum, %score : tensor<4xf64>
      %i_next = stablehlo.add %i, %one : tensor<i64>
      stablehlo.return %i_next, %next, %sum_next : tensor<i64>, tensor<4xf64>, tensor<4xf64>
    }
    return %r#2 : tensor<4xf64>
  }
  func.func @main(%a: tensor<4xf64>) -> tensor<4xf64> {
    %seed = stablehlo.constant dense<1.0> : tensor<4xf64>
    %gradient = enzyme.autodiff @loop(%a, %seed) <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    return %gradient : tensor<4xf64>
  }
}

// Both reverse reads share the original tape. The final state stays invariant.
// CHECK-LABEL: func.func private @diffeloop(
// CHECK: tensor<8x4xf64>
// CHECK: %[[FORWARD:[^ :]+]]:{{[0-9]+}} = stablehlo.while
// CHECK: stablehlo.while
// CHECK: stablehlo.dynamic_slice %[[FORWARD]]#{{[0-9]+}}
// CHECK: %[[NEXT:[^ ]+]] = stablehlo.add
// CHECK: %[[ROW:[^ ]+]] = stablehlo.dynamic_slice %[[FORWARD]]#{{[0-9]+}}, %[[NEXT]]
// CHECK: %[[CACHED:[^ ]+]] = stablehlo.reshape %[[ROW]]
// CHECK: %[[LAST:[^ ]+]] = stablehlo.compare EQ
// CHECK: %[[STATE:[^ ]+]] = stablehlo.select %[[LAST]], %[[FORWARD]]#1, %[[CACHED]]
// CHECK: stablehlo.divide {{.*}}, %[[STATE]]
