// RUN: enzymexlamlir-opt %s --enzyme --arith-raise="stablehlo=true" --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --cse | FileCheck %s

// The derivative rule reconstructs tanh from its input. The same result is
// already stored in the scan output, even though min-cut did not clone it.
module {
  func.func private @loop(%a: tensor<f64>) -> tensor<4xf64> {
    %zero = stablehlo.constant dense<0> : tensor<i64>
    %one = stablehlo.constant dense<1> : tensor<i64>
    %four = stablehlo.constant dense<4> : tensor<i64>
    %init = stablehlo.constant dense<0.2> : tensor<f64>
    %empty = stablehlo.constant dense<0.0> : tensor<4xf64>
    %r:3 = stablehlo.while(%i = %zero, %state = %init, %out = %empty) : tensor<i64>, tensor<f64>, tensor<4xf64>
    cond {
      %c = stablehlo.compare LT, %i, %four : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %product = stablehlo.multiply %a, %state : tensor<f64>
      %next = stablehlo.tanh %product : tensor<f64>
      %row = stablehlo.reshape %next : (tensor<f64>) -> tensor<1xf64>
      %updated = stablehlo.dynamic_update_slice %out, %row, %i : (tensor<4xf64>, tensor<1xf64>, tensor<i64>) -> tensor<4xf64>
      %i_next = stablehlo.add %i, %one : tensor<i64>
      stablehlo.return %i_next, %next, %updated : tensor<i64>, tensor<f64>, tensor<4xf64>
    }
    return %r#2 : tensor<4xf64>
  }
  func.func @main(%a: tensor<f64>) -> (tensor<4xf64>, tensor<f64>) {
    %seed = stablehlo.constant dense<1.0> : tensor<4xf64>
    %value, %gradient = enzyme.autodiff @loop(%a, %seed) <{activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_active>]}> : (tensor<f64>, tensor<4xf64>) -> (tensor<4xf64>, tensor<f64>)
    return %value, %gradient : tensor<4xf64>, tensor<f64>
  }
}

// CHECK-LABEL: func.func private @diffeloop(
// CHECK: %[[FORWARD:[^ :]+]]:{{[0-9]+}} = stablehlo.while
// CHECK: stablehlo.tanh
// CHECK: stablehlo.while
// CHECK: %[[ROW:[^ ]+]] = stablehlo.dynamic_slice %[[FORWARD]]#2
// CHECK: %[[VALUE:[^ ]+]] = stablehlo.reshape %[[ROW]]
// CHECK: stablehlo.multiply %[[VALUE]], %[[VALUE]]
// CHECK-NOT: stablehlo.tanh
// CHECK: return
