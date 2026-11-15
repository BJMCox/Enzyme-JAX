module {
  func.func @main(%a: tensor<2x2xf64>, %b: tensor<2x2xf64>) -> (tensor<2x2xf64>, tensor<i1>) {
    %x, %valid = func.call @__enzyme_linalg_solve_v1_matrix(%a, %b) : (tensor<2x2xf64>, tensor<2x2xf64>) -> (tensor<2x2xf64>, tensor<i1>)
    return %x, %valid : tensor<2x2xf64>, tensor<i1>
  }
  func.func private @__enzyme_linalg_solve_v1_matrix(%a: tensor<2x2xf64>, %b: tensor<2x2xf64>) -> (tensor<2x2xf64>, tensor<i1>) {
    %a00m = stablehlo.slice %a [0:1, 0:1] : (tensor<2x2xf64>) -> tensor<1x1xf64>
    %a01m = stablehlo.slice %a [0:1, 1:2] : (tensor<2x2xf64>) -> tensor<1x1xf64>
    %a10m = stablehlo.slice %a [1:2, 0:1] : (tensor<2x2xf64>) -> tensor<1x1xf64>
    %a11m = stablehlo.slice %a [1:2, 1:2] : (tensor<2x2xf64>) -> tensor<1x1xf64>
    %a00 = stablehlo.reshape %a00m : (tensor<1x1xf64>) -> tensor<f64>
    %a01 = stablehlo.reshape %a01m : (tensor<1x1xf64>) -> tensor<f64>
    %a10 = stablehlo.reshape %a10m : (tensor<1x1xf64>) -> tensor<f64>
    %a11 = stablehlo.reshape %a11m : (tensor<1x1xf64>) -> tensor<f64>
    %d0 = stablehlo.multiply %a00, %a11 : tensor<f64>
    %d1 = stablehlo.multiply %a01, %a10 : tensor<f64>
    %det = stablehlo.subtract %d0, %d1 : tensor<f64>
    %zero = stablehlo.constant dense<0.0> : tensor<f64>
    %one = stablehlo.constant dense<1.0> : tensor<f64>
    %inf = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
    %abs = stablehlo.abs %det : tensor<f64>
    %nonzero = stablehlo.compare NE, %det, %zero, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %finite = stablehlo.compare LT, %abs, %inf, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %valid = stablehlo.and %nonzero, %finite : tensor<i1>
    %safe = stablehlo.select %valid, %det, %one : tensor<i1>, tensor<f64>
    %b0 = stablehlo.slice %b [0:1, 0:2] : (tensor<2x2xf64>) -> tensor<1x2xf64>
    %b1 = stablehlo.slice %b [1:2, 0:2] : (tensor<2x2xf64>) -> tensor<1x2xf64>
    %c00 = stablehlo.broadcast_in_dim %a00, dims = [] : (tensor<f64>) -> tensor<1x2xf64>
    %c01 = stablehlo.broadcast_in_dim %a01, dims = [] : (tensor<f64>) -> tensor<1x2xf64>
    %c10 = stablehlo.broadcast_in_dim %a10, dims = [] : (tensor<f64>) -> tensor<1x2xf64>
    %c11 = stablehlo.broadcast_in_dim %a11, dims = [] : (tensor<f64>) -> tensor<1x2xf64>
    %p00 = stablehlo.multiply %c11, %b0 : tensor<1x2xf64>
    %p01 = stablehlo.multiply %c01, %b1 : tensor<1x2xf64>
    %p10 = stablehlo.multiply %c00, %b1 : tensor<1x2xf64>
    %p11 = stablehlo.multiply %c10, %b0 : tensor<1x2xf64>
    %r0 = stablehlo.subtract %p00, %p01 : tensor<1x2xf64>
    %r1 = stablehlo.subtract %p10, %p11 : tensor<1x2xf64>
    %rows = stablehlo.concatenate %r0, %r1, dim = 0 : (tensor<1x2xf64>, tensor<1x2xf64>) -> tensor<2x2xf64>
    %denom = stablehlo.broadcast_in_dim %safe, dims = [] : (tensor<f64>) -> tensor<2x2xf64>
    %result = stablehlo.divide %rows, %denom : tensor<2x2xf64>
    %pred = stablehlo.broadcast_in_dim %valid, dims = [] : (tensor<i1>) -> tensor<2x2xi1>
    %zeros = stablehlo.constant dense<0.0> : tensor<2x2xf64>
    %x = stablehlo.select %pred, %result, %zeros : tensor<2x2xi1>, tensor<2x2xf64>
    return %x, %valid : tensor<2x2xf64>, tensor<i1>
  }
}
