// RUN: enzymexlamlir-opt %s --split-input-file --xla-megakernelize --symbol-dce | FileCheck %s --implicit-check-not=scf.for

// A private function is not exclusive to the loop if another function calls it.
// Keep the original one-argument function for that caller. Give the lifted loop
// a separate function with a bound argument and a while loop.
module {
  func.func @lift_shared_body(%limit: i32, %data: memref<4xf32>) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    scf.for %iv = %zero to %limit step %one : i32 {
      enzymexla.xla_wrapper @shared_body(%data) : (memref<4xf32>) -> ()
    }
    return
  }

  func.func @call_shared_body(%data: tensor<4xf32>) -> tensor<4xf32> {
    %result = func.call @shared_body(%data) : (tensor<4xf32>) -> tensor<4xf32>
    return %result : tensor<4xf32>
  }

  func.func private @shared_body(%data: tensor<4xf32>) -> tensor<4xf32> {
    %result = stablehlo.negate %data : tensor<4xf32>
    return %result : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @lift_shared_body(
// CHECK: enzymexla.xla_wrapper @[[$SHARED_LOOP:rxla[$]megakernel_[0-9]+]]
// CHECK-SAME: : (memref<i32, 1>, memref<4xf32>) -> ()
// CHECK-LABEL: func.func @call_shared_body(
// CHECK-SAME: %[[CALL_DATA:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[CALL_RESULT:.*]] = call @shared_body(%[[CALL_DATA]]) : (tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: return %[[CALL_RESULT]] : tensor<4xf32>
// CHECK-LABEL: func.func private @shared_body(
// CHECK-SAME: %[[SHARED_DATA:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[SHARED_RESULT:.*]] = stablehlo.negate %[[SHARED_DATA]] : tensor<4xf32>
// CHECK-NEXT: return %[[SHARED_RESULT]] : tensor<4xf32>
// CHECK-NEXT: }
// CHECK: func.func private @[[$SHARED_LOOP]](
// CHECK-SAME: tensor<i32>, {{.*}}tensor<4xf32>) -> (tensor<i32>, tensor<4xf32>)
// CHECK: stablehlo.while
// CHECK: stablehlo.negate

// -----

// A public function can have callers outside this module. Even with one local
// use, keep its signature and body. Put the loop in a new private function.
module {
  func.func @lift_public_body(%limit: i32, %data: memref<4xf32>) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    scf.for %iv = %zero to %limit step %one : i32 {
      enzymexla.xla_wrapper @public_body(%data) : (memref<4xf32>) -> ()
    }
    return
  }

  func.func @public_body(%data: tensor<4xf32>) -> tensor<4xf32> {
    %result = stablehlo.negate %data : tensor<4xf32>
    return %result : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @lift_public_body(
// CHECK: enzymexla.xla_wrapper @[[$PUBLIC_LOOP:rxla[$]megakernel_[0-9]+]]
// CHECK-SAME: : (memref<i32, 1>, memref<4xf32>) -> ()
// CHECK-LABEL: func.func @public_body(
// CHECK-SAME: %[[PUBLIC_DATA:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[PUBLIC_RESULT:.*]] = stablehlo.negate %[[PUBLIC_DATA]] : tensor<4xf32>
// CHECK-NEXT: return %[[PUBLIC_RESULT]] : tensor<4xf32>
// CHECK-NEXT: }
// CHECK: func.func private @[[$PUBLIC_LOOP]](
// CHECK-SAME: tensor<i32>, {{.*}}tensor<4xf32>) -> (tensor<i32>, tensor<4xf32>)
// CHECK: stablehlo.while
// CHECK: stablehlo.negate

// -----

// A private function with only this wrapper use can change in place. Keep its
// name, add the bound argument/result, and replace its body with the while loop.
module {
  func.func @lift_exclusive_body(%limit: i32, %data: memref<4xf32>) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    scf.for %iv = %zero to %limit step %one : i32 {
      enzymexla.xla_wrapper @exclusive_body(%data) : (memref<4xf32>) -> ()
    }
    return
  }

  func.func private @exclusive_body(%data: tensor<4xf32>) -> tensor<4xf32> {
    %result = stablehlo.negate %data : tensor<4xf32>
    return %result : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @lift_exclusive_body(
// CHECK: enzymexla.xla_wrapper @exclusive_body
// CHECK-SAME: : (memref<i32, 1>, memref<4xf32>) -> ()
// CHECK-NOT: func.func private @rxla
// CHECK-LABEL: func.func private @exclusive_body(
// CHECK-SAME: %[[LIMIT:.*]]: tensor<i32>, %[[DATA:.*]]: tensor<4xf32>) -> (tensor<i32>, tensor<4xf32>)
// CHECK: %[[LOOP:.*]]:2 = stablehlo.while
// CHECK: stablehlo.negate
// CHECK: return %[[LIMIT]], %[[LOOP]]#1 : tensor<i32>, tensor<4xf32>
// CHECK-NOT: func.func private @rxla
