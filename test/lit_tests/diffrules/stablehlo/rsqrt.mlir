// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=rsqrt outfn= retTys=enzyme_dup argTys=enzyme_dup mode=ForwardMode" --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=FORWARD
// RUN: enzymexlamlir-opt %s --enzyme-wrap="infn=rsqrt outfn= retTys=enzyme_active argTys=enzyme_active mode=ReverseModeCombined" --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt --cse | FileCheck %s --check-prefix=REVERSE
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise --enzyme-hlo-opt | stablehlo-translate - --interpret --allow-unregistered-dialect

func.func @rsqrt(%x : tensor<2xf32>) -> tensor<2xf32> {
  %y = stablehlo.rsqrt %x : (tensor<2xf32>) -> tensor<2xf32>
  func.return %y : tensor<2xf32>
}

// FORWARD:    func.func @rsqrt(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
// FORWARD-NEXT:      %cst = stablehlo.constant dense<-5.000000e-01> : tensor<2xf32>
// FORWARD-NEXT:      %cst_0 = stablehlo.constant dense<-2.000000e+00> : tensor<2xf32>
// FORWARD-NEXT:      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
// FORWARD-NEXT:      %cst_2 = stablehlo.constant dense<2.500000e-01> : tensor<2xf32>
// FORWARD-NEXT:      %0 = stablehlo.compare GE, %arg0, %cst_2 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// FORWARD-NEXT:      %1 = stablehlo.compare LT, %arg0, %cst_1 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// FORWARD-NEXT:      %2 = stablehlo.and %0, %1 : tensor<2xi1>
// FORWARD-NEXT:      %3 = stablehlo.sqrt %arg0 : tensor<2xf32>
// FORWARD-NEXT:      %4 = stablehlo.multiply %arg0, %3 : tensor<2xf32>
// FORWARD-NEXT:      %5 = stablehlo.multiply %cst_0, %4 : tensor<2xf32>
// FORWARD-NEXT:      %6 = stablehlo.divide %arg1, %5 : tensor<2xf32>
// FORWARD-NEXT:      %7 = stablehlo.rsqrt %arg0 : tensor<2xf32>
// FORWARD-NEXT:      %8 = stablehlo.multiply %cst, %7 : tensor<2xf32>
// FORWARD-NEXT:      %9 = stablehlo.multiply %arg1, %8 : tensor<2xf32>
// FORWARD-NEXT:      %10 = stablehlo.divide %9, %arg0 : tensor<2xf32>
// FORWARD-NEXT:      %11 = stablehlo.select %2, %6, %10 : tensor<2xi1>, tensor<2xf32>
// FORWARD-NEXT:      return %7, %11 : tensor<2xf32>, tensor<2xf32>
// FORWARD-NEXT:    }

// REVERSE:    func.func @rsqrt(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
// REVERSE-NEXT:      %cst = stablehlo.constant dense<-5.000000e-01> : tensor<2xf32>
// REVERSE-NEXT:      %cst_0 = stablehlo.constant dense<-2.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:      %cst_1 = stablehlo.constant dense<1.000000e+00> : tensor<2xf32>
// REVERSE-NEXT:      %cst_2 = stablehlo.constant dense<2.500000e-01> : tensor<2xf32>
// REVERSE-NEXT:      %0 = stablehlo.compare GE, %arg0, %cst_2 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// REVERSE-NEXT:      %1 = stablehlo.compare LT, %arg0, %cst_1 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
// REVERSE-NEXT:      %2 = stablehlo.and %0, %1 : tensor<2xi1>
// REVERSE-NEXT:      %3 = stablehlo.sqrt %arg0 : tensor<2xf32>
// REVERSE-NEXT:      %4 = stablehlo.multiply %arg0, %3 : tensor<2xf32>
// REVERSE-NEXT:      %5 = stablehlo.multiply %cst_0, %4 : tensor<2xf32>
// REVERSE-NEXT:      %6 = stablehlo.divide %arg1, %5 : tensor<2xf32>
// REVERSE-NEXT:      %7 = stablehlo.rsqrt %arg0 : tensor<2xf32>
// REVERSE-NEXT:      %8 = stablehlo.multiply %cst, %7 : tensor<2xf32>
// REVERSE-NEXT:      %9 = stablehlo.multiply %arg1, %8 : tensor<2xf32>
// REVERSE-NEXT:      %10 = stablehlo.divide %9, %arg0 : tensor<2xf32>
// REVERSE-NEXT:      %11 = stablehlo.select %2, %6, %10 : tensor<2xi1>, tensor<2xf32>
// REVERSE-NEXT:      return %11 : tensor<2xf32>
// REVERSE-NEXT:    }

func.func @main() {
  %x = stablehlo.constant dense<[4.0, 9.0]> : tensor<2xf32>
  %out = stablehlo.constant dense<[0.5, 0.33333334]> : tensor<2xf32>
  %expected = stablehlo.constant dense<[-0.0625, -0.018518518518511842]> : tensor<2xf32>

  %d = stablehlo.constant dense<1.0> : tensor<2xf32>

  %fwd:2 = enzyme.fwddiff @rsqrt(%x, %d) <{
    activity=[#enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dup>]
  }> : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %fwd#0, %out : tensor<2xf32>
  check.expect_almost_eq %fwd#1, %expected : tensor<2xf32>

  %rev:2 = enzyme.autodiff @rsqrt(%x, %d) <{
    activity=[#enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_active>]
  }> : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>)

  check.expect_almost_eq %rev#0, %out : tensor<2xf32>
  check.expect_almost_eq %rev#1, %expected : tensor<2xf32>

  func.return
}
