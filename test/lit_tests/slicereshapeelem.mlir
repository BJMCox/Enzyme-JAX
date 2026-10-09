// RUN: enzymexlamlir-opt  --enzyme-hlo-generate-td="patterns=slice_reshape_elementwise" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @slice_sub(%x: tensor<1x1x8192x1x256xbf16>, %y: tensor<1x1x8192x1x256xbf16>) -> (tensor<3072x1x1x256xbf16>) {
  %1251 = stablehlo.subtract %x, %y : (tensor<1x1x8192x1x256xbf16>, tensor<1x1x8192x1x256xbf16>) -> tensor<1x1x8192x1x256xbf16> 
  %rs = stablehlo.reshape %1251 : (tensor<1x1x8192x1x256xbf16>) -> tensor<8192x1x1x256xbf16> 
  %1252 = stablehlo.slice %rs [0:3072, 0:1, 0:1, 0:256] : (tensor<8192x1x1x256xbf16>) -> tensor<3072x1x1x256xbf16>
  return %1252 : tensor<3072x1x1x256xbf16>
}

// CHECK:  func.func @slice_sub(%arg0: tensor<1x1x8192x1x256xbf16>, %arg1: tensor<1x1x8192x1x256xbf16>) -> tensor<3072x1x1x256xbf16> {
// CHECK-NEXT:    %0 = stablehlo.slice %arg0 [0:1, 0:1, 0:3072, 0:1, 0:256] : (tensor<1x1x8192x1x256xbf16>) -> tensor<1x1x3072x1x256xbf16>
// CHECK-NEXT:    %1 = stablehlo.slice %arg1 [0:1, 0:1, 0:3072, 0:1, 0:256] : (tensor<1x1x8192x1x256xbf16>) -> tensor<1x1x3072x1x256xbf16>
// CHECK-NEXT:    %2 = stablehlo.subtract %0, %1 : tensor<1x1x3072x1x256xbf16>
// CHECK-NEXT:    %3 = stablehlo.reshape %2 : (tensor<1x1x3072x1x256xbf16>) -> tensor<3072x1x1x256xbf16>
// CHECK-NEXT:    return %3 : tensor<3072x1x1x256xbf16>
// CHECK-NEXT:  }

func.func @strided_exp(%x: tensor<1x3xf32>) -> tensor<2xf32> {
  %y = stablehlo.exponential %x : tensor<1x3xf32>
  %r = stablehlo.reshape %y : (tensor<1x3xf32>) -> tensor<3xf32>
  %s = stablehlo.slice %r [0:3:2] : (tensor<3xf32>) -> tensor<2xf32>
  return %s : tensor<2xf32>
}

// CHECK-LABEL: func.func @strided_exp
// CHECK-NEXT: %[[S:.*]] = stablehlo.slice %arg0 [0:1, 0:3:2] : (tensor<1x3xf32>) -> tensor<1x2xf32>
// CHECK-NEXT: %[[E:.*]] = stablehlo.exponential %[[S]] : tensor<1x2xf32>
// CHECK-NEXT: %[[R:.*]] = stablehlo.reshape %[[E]] : (tensor<1x2xf32>) -> tensor<2xf32>
// CHECK-NEXT: return %[[R]] : tensor<2xf32>
