// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --enzyme-simplify-math --canonicalize --cse | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --lower-enzymexla-ml --inline --enzyme-hlo-opt --drop-unsupported-attributes --symbol-dce | stablehlo-translate --interpret

// The powers share one captured input. Recompute them in the active reverse
// branch instead of returning a tape tensor and a zero tensor for every power.
// Check both if branches and all three case branches against exact derivatives.
module {
  func.func private @if_power(%x: tensor<4xf64>, %pred: tensor<i1>) -> tensor<4xf64> {
    %y = "stablehlo.if"(%pred) ({
      %x2 = stablehlo.multiply %x, %x : tensor<4xf64>
      %x4 = stablehlo.multiply %x2, %x2 : tensor<4xf64>
      %x8 = stablehlo.multiply %x4, %x4 : tensor<4xf64>
      stablehlo.return %x8 : tensor<4xf64>
    }, {
      %x2 = stablehlo.multiply %x, %x : tensor<4xf64>
      %x3 = stablehlo.multiply %x2, %x : tensor<4xf64>
      stablehlo.return %x3 : tensor<4xf64>
    }) : (tensor<i1>) -> tensor<4xf64>
    return %y : tensor<4xf64>
  }

  func.func private @case_power(%x: tensor<4xf64>, %index: tensor<i32>) -> tensor<4xf64> {
    %y = "stablehlo.case"(%index) ({
      %x2 = stablehlo.multiply %x, %x : tensor<4xf64>
      %x4 = stablehlo.multiply %x2, %x2 : tensor<4xf64>
      %x8 = stablehlo.multiply %x4, %x4 : tensor<4xf64>
      stablehlo.return %x8 : tensor<4xf64>
    }, {
      %x2 = stablehlo.multiply %x, %x : tensor<4xf64>
      %x3 = stablehlo.multiply %x2, %x : tensor<4xf64>
      stablehlo.return %x3 : tensor<4xf64>
    }, {
      stablehlo.return %x : tensor<4xf64>
    }) : (tensor<i32>) -> tensor<4xf64>
    return %y : tensor<4xf64>
  }

  func.func private @nested_power(%x: tensor<4xf64>, %pred: tensor<i1>, %index: tensor<i32>) -> tensor<4xf64> {
    %y = "stablehlo.if"(%pred) ({
      %x2 = stablehlo.multiply %x, %x : tensor<4xf64>
      %v = "stablehlo.case"(%index) ({
        %x4 = stablehlo.multiply %x2, %x2 : tensor<4xf64>
        stablehlo.return %x4 : tensor<4xf64>
      }, {
        %x3 = stablehlo.multiply %x2, %x : tensor<4xf64>
        stablehlo.return %x3 : tensor<4xf64>
      }, {
        stablehlo.return %x2 : tensor<4xf64>
      }) : (tensor<i32>) -> tensor<4xf64>
      stablehlo.return %v : tensor<4xf64>
    }, {
      stablehlo.return %x : tensor<4xf64>
    }) : (tensor<i1>) -> tensor<4xf64>
    return %y : tensor<4xf64>
  }

  func.func @main() {
    %x = stablehlo.constant dense<[-1.0, 0.0, 1.0, 2.0]> : tensor<4xf64>
    %seed = stablehlo.constant dense<1.0> : tensor<4xf64>
    %power8 = stablehlo.constant dense<[-8.0, 0.0, 8.0, 1024.0]> : tensor<4xf64>
    %power3 = stablehlo.constant dense<[3.0, 0.0, 3.0, 12.0]> : tensor<4xf64>
    %true = stablehlo.constant dense<true> : tensor<i1>
    %false = stablehlo.constant dense<false> : tensor<i1>
    %c0 = stablehlo.constant dense<0> : tensor<i32>
    %c1 = stablehlo.constant dense<1> : tensor<i32>
    %c2 = stablehlo.constant dense<2> : tensor<i32>
    %a = enzyme.autodiff @if_power(%x, %true, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (tensor<4xf64>, tensor<i1>, tensor<4xf64>) -> tensor<4xf64>
    %b = enzyme.autodiff @if_power(%x, %false, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (tensor<4xf64>, tensor<i1>, tensor<4xf64>) -> tensor<4xf64>
    %c = enzyme.autodiff @case_power(%x, %c0, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (tensor<4xf64>, tensor<i32>, tensor<4xf64>) -> tensor<4xf64>
    %d = enzyme.autodiff @case_power(%x, %c1, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (tensor<4xf64>, tensor<i32>, tensor<4xf64>) -> tensor<4xf64>
    %e = enzyme.autodiff @case_power(%x, %c2, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (tensor<4xf64>, tensor<i32>, tensor<4xf64>) -> tensor<4xf64>
    %f = enzyme.autodiff @nested_power(%x, %true, %c1, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_const>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (tensor<4xf64>, tensor<i1>, tensor<i32>, tensor<4xf64>) -> tensor<4xf64>
    %g = enzyme.autodiff @nested_power(%x, %false, %c1, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_const>, #enzyme.activity<enzyme_const>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (tensor<4xf64>, tensor<i1>, tensor<i32>, tensor<4xf64>) -> tensor<4xf64>
    check.expect_close %a, %power8, max_ulp_difference = 0, min_ulp_difference = 0 : tensor<4xf64>, tensor<4xf64>
    check.expect_close %b, %power3, max_ulp_difference = 0, min_ulp_difference = 0 : tensor<4xf64>, tensor<4xf64>
    check.expect_close %c, %power8, max_ulp_difference = 0, min_ulp_difference = 0 : tensor<4xf64>, tensor<4xf64>
    check.expect_close %d, %power3, max_ulp_difference = 0, min_ulp_difference = 0 : tensor<4xf64>, tensor<4xf64>
    check.expect_close %e, %seed, max_ulp_difference = 0, min_ulp_difference = 0 : tensor<4xf64>, tensor<4xf64>
    check.expect_close %f, %power3, max_ulp_difference = 0, min_ulp_difference = 0 : tensor<4xf64>, tensor<4xf64>
    check.expect_close %g, %seed, max_ulp_difference = 0, min_ulp_difference = 0 : tensor<4xf64>, tensor<4xf64>
    return
  }
}

// One reverse branch remains. The forward branch no longer returns its tape.
// CHECK-LABEL: func.func private @diffeif_power(
// CHECK: "stablehlo.if"
// CHECK: }) : (tensor<i1>) ->
// CHECK-NOT: "stablehlo.if"
// CHECK: return
// CHECK-LABEL: func.func private @diffecase_power(
// CHECK: "stablehlo.case"
// CHECK: }) : (tensor<i32>) ->
// CHECK-NOT: "stablehlo.case"
// CHECK: return
