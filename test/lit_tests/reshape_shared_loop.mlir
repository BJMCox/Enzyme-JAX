// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=reshape_elementwise(0)" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

// A recurrence and its output row must share the same arithmetic.
// CHECK-LABEL: func.func @shared_loop
// CHECK: %[[SQUARE:.*]] = stablehlo.multiply
// CHECK: %[[NEXT:.*]] = stablehlo.add %[[SQUARE]],
// CHECK-NEXT: %[[ROW:.*]] = stablehlo.reshape %[[NEXT]]
// CHECK: stablehlo.return {{.*}}, %[[NEXT]], %[[ROW]]
func.func @shared_loop(%x: tensor<4xf64>) -> (tensor<4xf64>, tensor<4x1xf64>) {
  %zero = stablehlo.constant dense<0> : tensor<i64>
  %one = stablehlo.constant dense<1> : tensor<i64>
  %end = stablehlo.constant dense<16> : tensor<i64>
  %empty = stablehlo.constant dense<0.0> : tensor<4x1xf64>
  %result:3 = stablehlo.while(%i = %zero, %state = %x, %out = %empty) : tensor<i64>, tensor<4xf64>, tensor<4x1xf64>
  cond {
    %more = stablehlo.compare LT, %i, %end : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %more : tensor<i1>
  } do {
    %square = stablehlo.multiply %state, %state : tensor<4xf64>
    %next = stablehlo.add %square, %x : tensor<4xf64>
    %row = stablehlo.reshape %next : (tensor<4xf64>) -> tensor<4x1xf64>
    %inc = stablehlo.add %i, %one : tensor<i64>
    stablehlo.return %inc, %next, %row : tensor<i64>, tensor<4xf64>, tensor<4x1xf64>
  }
  return %result#1, %result#2 : tensor<4xf64>, tensor<4x1xf64>
}

// A single-use suffix still moves, but its shared input stays shared.
// CHECK-LABEL: func.func @shared_intermediate
// CHECK: %[[SHARED:.*]] = stablehlo.multiply {{.*}} : tensor<4xf64>
// CHECK-NEXT: %[[ROW:.*]] = stablehlo.reshape %[[SHARED]]
// CHECK-NEXT: %[[EXP:.*]] = stablehlo.exponential %[[ROW]] : tensor<4x1xf64>
// CHECK: stablehlo.return {{.*}}, %[[SHARED]], %[[EXP]]
func.func @shared_intermediate(%x: tensor<4xf64>) -> (tensor<4xf64>, tensor<4x1xf64>) {
  %zero = stablehlo.constant dense<0> : tensor<i64>
  %one = stablehlo.constant dense<1> : tensor<i64>
  %end = stablehlo.constant dense<16> : tensor<i64>
  %empty = stablehlo.constant dense<0.0> : tensor<4x1xf64>
  %result:3 = stablehlo.while(%i = %zero, %state = %x, %out = %empty) : tensor<i64>, tensor<4xf64>, tensor<4x1xf64>
  cond {
    %more = stablehlo.compare LT, %i, %end : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %more : tensor<i1>
  } do {
    %shared = stablehlo.multiply %state, %x : tensor<4xf64>
    %output = stablehlo.exponential %shared : tensor<4xf64>
    %row = stablehlo.reshape %output : (tensor<4xf64>) -> tensor<4x1xf64>
    %inc = stablehlo.add %i, %one : tensor<i64>
    stablehlo.return %inc, %shared, %row : tensor<i64>, tensor<4xf64>, tensor<4x1xf64>
  }
  return %result#1, %result#2 : tensor<4xf64>, tensor<4x1xf64>
}

// Aggressive propagation outside loops remains available.
// CHECK-LABEL: func.func @outside_loop
// CHECK: stablehlo.multiply {{.*}} : tensor<4x1xf64>
// CHECK: stablehlo.multiply {{.*}} : tensor<4xf64>
func.func @outside_loop(%x: tensor<4xf64>) -> (tensor<4xf64>, tensor<4x1xf64>) {
  %shared = stablehlo.multiply %x, %x : tensor<4xf64>
  %row = stablehlo.reshape %shared : (tensor<4xf64>) -> tensor<4x1xf64>
  return %shared, %row : tensor<4xf64>, tensor<4x1xf64>
}
