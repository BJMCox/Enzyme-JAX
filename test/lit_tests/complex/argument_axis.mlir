// RUN: enzymexlamlir-opt --lower-complex-operations="concat-dimension=first" --canonicalize %s | FileCheck %s --check-prefix=FIRST --implicit-check-not=unrealized_conversion_cast
// RUN: enzymexlamlir-opt --lower-complex-operations="concat-dimension=last" --canonicalize %s | FileCheck %s --check-prefix=LAST --implicit-check-not=unrealized_conversion_cast

// FIRST-LABEL: @twice
// FIRST: stablehlo.concatenate {{.*}} dim = 0 : (tensor<1x3x2xf64>, tensor<1x3x2xf64>) -> tensor<2x3x2xf64>
// FIRST: return {{.*}} : tensor<3x2xcomplex<f64>>
// LAST-LABEL: @twice
// LAST: stablehlo.concatenate {{.*}} dim = 2 : (tensor<3x2x1xf64>, tensor<3x2x1xf64>) -> tensor<3x2x2xf64>
// LAST: return {{.*}} : tensor<3x2xcomplex<f64>>
func.func @twice(%arg: tensor<3x2xcomplex<f64>>) -> tensor<3x2xcomplex<f64>> {
  %res = stablehlo.add %arg, %arg : tensor<3x2xcomplex<f64>>
  return %res : tensor<3x2xcomplex<f64>>
}
