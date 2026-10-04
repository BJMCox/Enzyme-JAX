// RUN: enzymexlamlir-opt %s --xla-megakernelize --symbol-dce | FileCheck %s --implicit-check-not=scf.for --implicit-check-not=gpu.alloc --implicit-check-not=gpu.dealloc

// Fuse two calls, then lift their loop. Keep the limit, step, and specialized
// scalar outside the while state. Put stack slots in the function entry block.
// Keep stores and copies inside the host guard.
module {
  func.func @guarded_specialized_loop(%limit: i64, %step: i64,
                                     %data: memref<i32>, %scale: i32) {
    %zero = arith.constant 0 : i64
    %positive = arith.cmpi sgt, %step, %zero : i64
    scf.if %positive {
      scf.for %iv = %zero to %limit step %step : i64 {
        enzymexla.xla_wrapper @add_scale(%data, %scale)
            <{num_specialized = 1 : i64}> : (memref<i32>, i32) -> ()
        enzymexla.xla_wrapper @negate(%data) : (memref<i32>) -> ()
      }
    }
    return
  }

  func.func private @add_scale(%data: tensor<i32>, %scale: tensor<i32>)
      -> tensor<i32> {
    %sum = stablehlo.add %data, %scale : tensor<i32>
    return %sum : tensor<i32>
  }

  func.func private @negate(%data: tensor<i32>) -> tensor<i32> {
    %negated = stablehlo.negate %data : tensor<i32>
    return %negated : tensor<i32>
  }
}

// CHECK-LABEL: func.func @guarded_specialized_loop(
// CHECK-SAME: %[[LIMIT:.*]]: i64, %[[STEP:.*]]: i64, %[[DATA:.*]]: memref<i32>, %[[SCALE:.*]]: i32)
// CHECK-NEXT: %[[BYTES:.*]] = arith.constant 8 : index
// CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK-NEXT: %[[STEP_HOST:.*]] = memref.alloca() : memref<i64>
// CHECK-NEXT: %[[LIMIT_HOST:.*]] = memref.alloca() : memref<i64>
// CHECK-NEXT: %[[POSITIVE:.*]] = arith.cmpi sgt, %[[STEP]], %[[ZERO]] : i64
// CHECK-NEXT: scf.if %[[POSITIVE]] {
// CHECK-NEXT: memref.store %[[LIMIT]], %[[LIMIT_HOST]][] : memref<i64>
// CHECK-NEXT: %[[LIMIT_DEVICE:.*]] = enzymexla.get_global_temp @[[LIMIT_TEMP:.*]] : memref<i64, 1>
// CHECK-NEXT: enzymexla.memcpy %[[LIMIT_DEVICE]], %[[LIMIT_HOST]], %[[BYTES]] : memref<i64, 1>, memref<i64>
// CHECK-NEXT: memref.store %[[STEP]], %[[STEP_HOST]][] : memref<i64>
// CHECK-NEXT: %[[STEP_DEVICE:.*]] = enzymexla.get_global_temp @[[STEP_TEMP:.*]] : memref<i64, 1>
// CHECK-NEXT: enzymexla.memcpy %[[STEP_DEVICE]], %[[STEP_HOST]], %[[BYTES]] : memref<i64, 1>, memref<i64>
// CHECK-NEXT: enzymexla.xla_wrapper @[[KERNEL:add_scale]] (%[[LIMIT_DEVICE]], %[[STEP_DEVICE]], %[[DATA]], %[[SCALE]]) <num_specialized = 1> : (memref<i64, 1>, memref<i64, 1>, memref<i32>, i32) -> ()
// CHECK-NEXT: }
// CHECK-NEXT: return
// CHECK-NEXT: }

// The specialized scalar is the last input. Only the bound buffers and updated
// data have results. The bound buffers return their original values.
// CHECK: func.func private @[[KERNEL]](
// CHECK-SAME: %[[LIMIT:.*]]: tensor<i64>, %[[STEP:.*]]: tensor<i64>, %[[DATA:.*]]: tensor<i32>, %[[SCALE:.*]]: tensor<i32>) -> (tensor<i64>, tensor<i64>, tensor<i32>)
// CHECK-NEXT: %[[START:.*]] = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT: %[[RESULTS:.*]]:2 = stablehlo.while(%[[IV:.*]] = %[[START]], %[[STATE:.*]] = %[[DATA]]) : tensor<i64>, tensor<i32>
// CHECK-NEXT: cond {
// CHECK-NEXT: %[[CONTINUE:.*]] = stablehlo.compare LT, %[[IV]], %[[LIMIT]], SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
// CHECK-NEXT: stablehlo.return %[[CONTINUE]] : tensor<i1>
// CHECK-NEXT: } do {
// CHECK-NEXT: %[[NEXT:.*]] = stablehlo.add %[[IV]], %[[STEP]] : tensor<i64>
// CHECK-NEXT: %[[SUM:.*]] = stablehlo.add %[[STATE]], %[[SCALE]] : tensor<i32>
// CHECK-NEXT: %[[NEGATED:.*]] = stablehlo.negate %[[SUM]] : tensor<i32>
// CHECK-NEXT: stablehlo.return %[[NEXT]], %[[NEGATED]] : tensor<i64>, tensor<i32>
// CHECK-NEXT: }
// CHECK-NEXT: return %[[LIMIT]], %[[STEP]], %[[RESULTS]]#1 : tensor<i64>, tensor<i64>, tensor<i32>
// CHECK-NEXT: }
// CHECK-NEXT: enzymexla.temp_alloc "private" @[[LIMIT_TEMP]] : memref<i64, 1>
// CHECK-NEXT: enzymexla.temp_alloc "private" @[[STEP_TEMP]] : memref<i64, 1>
// CHECK-NEXT: }
