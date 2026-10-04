// RUN: enzymexlamlir-opt %s --xla-megakernelize --symbol-dce | FileCheck %s

// Before: a guarded loop calls the wrapper inside an explicit allocation scope.
// After: the bound's stack slot is at the start of that scope. The store, device
// copy, and wrapper call stay inside the guard. Do not move the slot to the
// enclosing function or allocate it inside the guard.
module {
  func.func @nested_allocation_scope(%run: i1, %limit: i64, %data: memref<i32>) {
    memref.alloca_scope {
      scf.if %run {
        %zero = arith.constant 0 : i64
        %one = arith.constant 1 : i64
        scf.for %iv = %zero to %limit step %one : i64 {
          enzymexla.xla_wrapper @nested_body(%data) : (memref<i32>) -> ()
        }
      }
      memref.alloca_scope.return
    }
    return
  }

  func.func private @nested_body(%data: tensor<i32>) -> tensor<i32> {
    %negated = stablehlo.negate %data : tensor<i32>
    return %negated : tensor<i32>
  }
}

// CHECK-LABEL: func.func @nested_allocation_scope(
// CHECK-SAME: %[[GUARD:.*]]: i1, %[[LIMIT:.*]]: i64, %[[DATA:.*]]: memref<i32>)
// CHECK-NOT: memref.alloca()
// CHECK: memref.alloca_scope {
// CHECK-NEXT: %[[HOST:.*]] = memref.alloca() : memref<i64>
// CHECK-NEXT: scf.if %[[GUARD]] {
// CHECK-NEXT: memref.store %[[LIMIT]], %[[HOST]][] : memref<i64>
// CHECK-NEXT: %[[DEVICE:.*]] = enzymexla.get_global_temp @nested_body_bound_1 : memref<i64, 1>
// CHECK-NEXT: enzymexla.memcpy %[[DEVICE]], %[[HOST]], %{{.*}} : memref<i64, 1>, memref<i64>
// CHECK-NEXT: enzymexla.xla_wrapper @nested_body (%[[DEVICE]], %[[DATA]]) : (memref<i64, 1>, memref<i32>) -> ()
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: return
// CHECK-NEXT: }

// CHECK-LABEL: func.func private @nested_body(
// CHECK-SAME: %[[BOUND:.*]]: tensor<i64>, %[[INITIAL:.*]]: tensor<i32>) -> (tensor<i64>, tensor<i32>)
// CHECK: stablehlo.while
// CHECK: stablehlo.negate
