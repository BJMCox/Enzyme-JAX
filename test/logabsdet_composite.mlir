module {
  func.func @main(%a: tensor<2x2xf64>) -> (tensor<f64>, tensor<2x2xf64>, tensor<i1>) {
    %value, %inverse, %valid = func.call @__enzyme_linalg_logabsdet_v1_matrix(%a) : (tensor<2x2xf64>) -> (tensor<f64>, tensor<2x2xf64>, tensor<i1>)
    return %value, %inverse, %valid : tensor<f64>, tensor<2x2xf64>, tensor<i1>
  }
  func.func private @__enzyme_linalg_logabsdet_v1_matrix(%a: tensor<2x2xf64>) -> (tensor<f64>, tensor<2x2xf64>, tensor<i1>) {
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
    %magnitude = stablehlo.abs %det : tensor<f64>
    %value = stablehlo.log %magnitude : tensor<f64>
    %zero = stablehlo.constant dense<0.0> : tensor<f64>
    %one = stablehlo.constant dense<1.0> : tensor<f64>
    %inf = stablehlo.constant dense<0x7FF0000000000000> : tensor<f64>
    %nonzero = stablehlo.compare NE, %det, %zero, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %finite = stablehlo.compare LT, %magnitude, %inf, FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %valid = stablehlo.and %nonzero, %finite : tensor<i1>
    %safe = stablehlo.select %valid, %det, %one : tensor<i1>, tensor<f64>
    %reversed = stablehlo.reverse %a, dims = [0, 1] : tensor<2x2xf64>
    %transposed = stablehlo.transpose %reversed, dims = [1, 0] : (tensor<2x2xf64>) -> tensor<2x2xf64>
    %signs = stablehlo.constant dense<[[1.0, -1.0], [-1.0, 1.0]]> : tensor<2x2xf64>
    %cofactors = stablehlo.multiply %transposed, %signs : tensor<2x2xf64>
    %denom = stablehlo.broadcast_in_dim %safe, dims = [] : (tensor<f64>) -> tensor<2x2xf64>
    %result = stablehlo.divide %cofactors, %denom : tensor<2x2xf64>
    %pred = stablehlo.broadcast_in_dim %valid, dims = [] : (tensor<i1>) -> tensor<2x2xi1>
    %nan = stablehlo.constant dense<0x7FF8000000000000> : tensor<2x2xf64>
    %inverse = stablehlo.select %pred, %result, %nan : tensor<2x2xi1>, tensor<2x2xf64>
    return %value, %inverse, %valid : tensor<f64>, tensor<2x2xf64>, tensor<i1>
  }
}
