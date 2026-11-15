module {
  func.func @main(%a: tensor<1x1xf64>, %b: tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<i1>) {
    %x, %valid = func.call @__enzyme_linalg_solve_v1_test(%a, %b) : (tensor<1x1xf64>, tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<i1>)
    return %x, %valid : tensor<1x1xf64>, tensor<i1>
  }

  func.func private @identity(%x: tensor<1x1xf64>) -> tensor<1x1xf64> {
    return %x : tensor<1x1xf64>
  }

  func.func private @__enzyme_linalg_solve_v1_test(%a: tensor<1x1xf64>, %b: tensor<1x1xf64>) -> (tensor<1x1xf64>, tensor<i1>) {
    %av = stablehlo.reshape %a : (tensor<1x1xf64>) -> tensor<f64>
    %bv = stablehlo.reshape %b : (tensor<1x1xf64>) -> tensor<f64>
    %zero = stablehlo.constant dense<0.0> : tensor<f64>
    %one = stablehlo.constant dense<1.0> : tensor<f64>
    %inf = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
    %abs = stablehlo.abs %av : tensor<f64>
    %nonzero = stablehlo.compare NE, %av, %zero, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %finite = stablehlo.compare LT, %abs, %inf, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %valid = stablehlo.and %nonzero, %finite : tensor<i1>
    %safea = stablehlo.select %valid, %av, %one : tensor<i1>, tensor<f64>
    %safeb = stablehlo.select %valid, %bv, %zero : tensor<i1>, tensor<f64>
    %v = stablehlo.divide %safeb, %safea : tensor<f64>
    %x = stablehlo.reshape %v : (tensor<f64>) -> tensor<1x1xf64>
    return %x, %valid : tensor<1x1xf64>, tensor<i1>
  }
}
