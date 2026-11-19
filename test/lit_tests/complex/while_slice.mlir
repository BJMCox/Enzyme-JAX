// RUN: enzymexlamlir-opt --lower-complex-operations="concat-dimension=first" --canonicalize %s | FileCheck %s --check-prefix=FIRST --implicit-check-not=unrealized_conversion_cast
// RUN: enzymexlamlir-opt --lower-complex-operations="concat-dimension=last" --canonicalize %s | FileCheck %s --check-prefix=LAST --implicit-check-not=unrealized_conversion_cast

// FIRST-LABEL: @indexed_loop
// FIRST: %[[Z:.*]] = stablehlo.constant dense<0> : tensor<i64>
// FIRST: stablehlo.while(%[[I:[^ ]+]] = %[[Z]], {{.*}} : tensor<i64>, tensor<2x3x4xf32>
// FIRST: stablehlo.dynamic_slice {{[^,]+}}, %[[Z]], %[[I]], %[[Z]], sizes = [2, 1, 3]
// FIRST: stablehlo.dynamic_update_slice {{[^,]+}}, {{[^,]+}}, %[[Z]], %[[Z]], %[[I]] :
// FIRST: stablehlo.return {{.*}} : tensor<i64>, tensor<2x3x4xf32>
// LAST-LABEL: @indexed_loop
// LAST: %[[Z:.*]] = stablehlo.constant dense<0> : tensor<i64>
// LAST: stablehlo.while(%[[I:[^ ]+]] = %[[Z]], {{.*}} : tensor<i64>, tensor<3x4x2xf32>
// LAST: stablehlo.dynamic_slice {{[^,]+}}, %[[I]], %[[Z]], %[[Z]], sizes = [1, 3, 2]
// LAST: stablehlo.dynamic_update_slice {{[^,]+}}, {{[^,]+}}, %[[Z]], %[[I]], %[[Z]] :
// LAST: stablehlo.return {{.*}} : tensor<i64>, tensor<3x4x2xf32>
func.func @indexed_loop(%re: tensor<3x4xf32>, %im: tensor<3x4xf32>, %n: tensor<i64>) -> tensor<3x4xf32> {
  %z = stablehlo.complex %re, %im : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xcomplex<f32>>
  %zero = stablehlo.constant dense<0> : tensor<i64>
  %one = stablehlo.constant dense<1> : tensor<i64>
  %out:2 = stablehlo.while(%i = %zero, %state = %z) : tensor<i64>, tensor<3x4xcomplex<f32>>
  cond {
    %keep = stablehlo.compare LT, %i, %n, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
    stablehlo.return %keep : tensor<i1>
  } do {
    %real = stablehlo.real %state : (tensor<3x4xcomplex<f32>>) -> tensor<3x4xf32>
    %imag = stablehlo.imag %state : (tensor<3x4xcomplex<f32>>) -> tensor<3x4xf32>
    %swapped = stablehlo.complex %imag, %real : tensor<3x4xcomplex<f32>>
    %v = stablehlo.dynamic_slice %swapped, %i, %zero, sizes = [1, 3] : (tensor<3x4xcomplex<f32>>, tensor<i64>, tensor<i64>) -> tensor<1x3xcomplex<f32>>
    %next = stablehlo.dynamic_update_slice %state, %v, %zero, %i : (tensor<3x4xcomplex<f32>>, tensor<1x3xcomplex<f32>>, tensor<i64>, tensor<i64>) -> tensor<3x4xcomplex<f32>>
    %j = stablehlo.add %i, %one : tensor<i64>
    stablehlo.return %j, %next : tensor<i64>, tensor<3x4xcomplex<f32>>
  }
  %r = stablehlo.real %out#1 : (tensor<3x4xcomplex<f32>>) -> tensor<3x4xf32>
  return %r : tensor<3x4xf32>
}
