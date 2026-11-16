module {
  func.func private @single(tensor<2x2xf64>, tensor<2x2xf64>) -> (tensor<2x2xf64>, tensor<i1>)
  func.func @main(%a: tensor<2x2xf64>, %b: tensor<2x2xf64>, %weight: tensor<2x2xf64>) -> tensor<f64> {
    %zero = stablehlo.constant dense<0> : tensor<i64>
    %one = stablehlo.constant dense<1> : tensor<i64>
    %steps = stablehlo.constant dense<32> : tensor<i64>
    %init = stablehlo.constant dense<0.0> : tensor<f64>
    %epsilon = stablehlo.constant dense<0.001> : tensor<f64>
    %eye = stablehlo.constant dense<[[1.0, 0.0], [0.0, 1.0]]> : tensor<2x2xf64>
    %result:2 = stablehlo.while(%index = %zero, %total = %init) : tensor<i64>, tensor<f64>
    cond {
      %continue = stablehlo.compare LT, %index, %steps, SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %continue : tensor<i1>
    } do {
      %i = stablehlo.convert %index : (tensor<i64>) -> tensor<f64>
      %scale = stablehlo.multiply %i, %epsilon : tensor<f64>
      %full = stablehlo.broadcast_in_dim %scale, dims = [] : (tensor<f64>) -> tensor<2x2xf64>
      %delta = stablehlo.multiply %full, %eye : tensor<2x2xf64>
      %matrix = stablehlo.add %a, %delta : tensor<2x2xf64>
      %x, %valid = func.call @single(%matrix, %b) : (tensor<2x2xf64>, tensor<2x2xf64>) -> (tensor<2x2xf64>, tensor<i1>)
      %product = stablehlo.multiply %x, %weight : tensor<2x2xf64>
      %sum = stablehlo.reduce(%product init: %init) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x2xf64>, tensor<f64>) -> tensor<f64>
      %next = stablehlo.add %index, %one : tensor<i64>
      %value = stablehlo.add %total, %sum : tensor<f64>
      stablehlo.return %next, %value : tensor<i64>, tensor<f64>
    }
    return %result#1 : tensor<f64>
  }
}
