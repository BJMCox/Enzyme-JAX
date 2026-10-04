// RUN: enzymexlamlir-opt %s --split-input-file --xla-megakernelize --symbol-dce | FileCheck %s --implicit-check-not=gpu.alloc --implicit-check-not=gpu.dealloc
// RUN: enzymexlamlir-opt %s --split-input-file --xla-megakernelize --symbol-dce --enzyme-hlo-unroll=max-num-iterations=4 | FileCheck %s --check-prefix=UNROLL

module {
  llvm.func @lift_dynamic_loop(%lb: i32, %ub: i32, %step: i32,
                              %arg0: !llvm.ptr {llvm.noalias},
                              %arg1: !llvm.ptr {llvm.noalias}) {
    scf.for unsigned %iv = %lb to %ub step %step : i32 {
      %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf32>
      %1 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xf32>
      enzymexla.xla_wrapper @loop_body (%0, %1) :
          (memref<?xf32>, memref<?xf32>) -> ()
    }
    llvm.return
  }

  func.func private @loop_body(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<?xf32>
    return %arg0, %0 : tensor<?xf32>, tensor<?xf32>
  }
}

// -----

module {
  llvm.func @lift_static_loop(%arg0: !llvm.ptr {llvm.noalias},
                             %arg1: !llvm.ptr {llvm.noalias}) {
    %c1_i32 = arith.constant 1 : i32
    %c5_i32 = arith.constant 5 : i32
    %c2_i32 = arith.constant 2 : i32
    scf.for %iv = %c1_i32 to %c5_i32 step %c2_i32 : i32 {
      %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<4xf32>
      %1 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<4xf32>
      enzymexla.xla_wrapper @static_loop_body (%0, %1) :
          (memref<4xf32>, memref<4xf32>) -> ()
    }
    llvm.return
  }

  func.func private @static_loop_body(%arg0: tensor<4xf32>,
                                       %arg1: tensor<4xf32>)
      -> (tensor<4xf32>, tensor<4xf32>) {
    %0 = stablehlo.subtract %arg0, %arg1 : tensor<4xf32>
    return %0, %arg1 : tensor<4xf32>, tensor<4xf32>
  }
}

// -----

// Loop lifting reuses the existing wrapper's buffer arguments.
// It needs no additional alias proof.
module {
  llvm.func @lift_without_noalias(%lb: i32, %ub: i32, %step: i32,
                                    %arg0: !llvm.ptr, %arg1: !llvm.ptr) {
    scf.for %iv = %lb to %ub step %step : i32 {
      %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf32>
      %1 = "enzymexla.pointer2memref"(%arg1) : (!llvm.ptr) -> memref<?xf32>
      enzymexla.xla_wrapper @may_alias_loop_body (%0, %1) :
          (memref<?xf32>, memref<?xf32>) -> ()
    }
    llvm.return
  }

  func.func private @may_alias_loop_body(%arg0: tensor<?xf32>,
                                           %arg1: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>) {
    %0 = stablehlo.subtract %arg0, %arg1 : tensor<?xf32>
    return %arg0, %0 : tensor<?xf32>, tensor<?xf32>
  }
}

// CHECK-LABEL: llvm.func @lift_dynamic_loop(
// CHECK-SAME:    %[[HOST_LB:.*]]: i32, %[[HOST_UB:.*]]: i32, %[[HOST_STEP:.*]]: i32,
// CHECK-SAME:    %[[HOST_BUFFER0:.*]]: !llvm.ptr {{.*}}, %[[HOST_BUFFER1:.*]]: !llvm.ptr
// CHECK-NOT:     scf.for
// CHECK:         %[[C4:.*]] = arith.constant 4 : index
// CHECK-COUNT-3: memref.alloca() : memref<i32>
// CHECK:         memref.store %[[HOST_LB]], %[[LB_HOST:.*]][] : memref<i32>
// CHECK:         %[[LB_DEVICE:.*]] = enzymexla.get_global_temp @[[LB_TEMP:loop_body_bound_0]] : memref<i32, 1>
// CHECK:         enzymexla.memcpy %[[LB_DEVICE]], %[[LB_HOST]], %[[C4]] : memref<i32, 1>, memref<i32>
// CHECK:         memref.store %[[HOST_UB]], %[[UB_HOST:.*]][] : memref<i32>
// CHECK:         %[[UB_DEVICE:.*]] = enzymexla.get_global_temp @[[UB_TEMP:loop_body_bound_1]] : memref<i32, 1>
// CHECK:         enzymexla.memcpy %[[UB_DEVICE]], %[[UB_HOST]], %[[C4]] : memref<i32, 1>, memref<i32>
// CHECK:         memref.store %[[HOST_STEP]], %[[STEP_HOST:.*]][] : memref<i32>
// CHECK:         %[[STEP_DEVICE:.*]] = enzymexla.get_global_temp @[[STEP_TEMP:loop_body_bound_2]] : memref<i32, 1>
// CHECK:         enzymexla.memcpy %[[STEP_DEVICE]], %[[STEP_HOST]], %[[C4]] : memref<i32, 1>, memref<i32>
// CHECK:         %[[ARG0_MEMREF:.*]] = "enzymexla.pointer2memref"(%[[HOST_BUFFER0]])
// CHECK:         %[[ARG1_MEMREF:.*]] = "enzymexla.pointer2memref"(%[[HOST_BUFFER1]])
// CHECK:         enzymexla.xla_wrapper @[[SCF_KERNEL:loop_body]]
// CHECK-SAME:      (%[[LB_DEVICE]], %[[UB_DEVICE]], %[[STEP_DEVICE]], %[[ARG0_MEMREF]], %[[ARG1_MEMREF]])
// CHECK:         llvm.return

// Reuse the private function because this wrapper is its only caller.
// CHECK:       func.func private @[[SCF_KERNEL]](
// CHECK-SAME:      %[[LB:.*]]: tensor<i32>, %[[UB:.*]]: tensor<i32>, %[[STEP:.*]]: tensor<i32>
// CHECK-SAME:      %[[BUFFER0:.*]]: tensor<?xf32>, %[[BUFFER1:.*]]: tensor<?xf32>
// The limit and step stay in the entry block. Carry only the IV and buffers.
// CHECK:         %[[RESULTS:.*]]:3 = stablehlo.while(%[[ITER:.*]] = %[[LB]], %[[STATE0:.*]] = %[[BUFFER0]], %[[STATE1:.*]] = %[[BUFFER1]]) : tensor<i32>, tensor<?xf32>, tensor<?xf32>
// CHECK:         cond {
// CHECK:           %[[CONTINUE:.*]] = stablehlo.compare LT, %[[ITER]], %[[UB]], UNSIGNED
// CHECK:           stablehlo.return %[[CONTINUE]]
// CHECK:         } do {
// CHECK:           %[[NEXT:.*]] = stablehlo.add %[[ITER]], %[[STEP]]
// CHECK:           %[[UPDATED:.*]] = stablehlo.add %[[STATE0]], %[[STATE1]]
// CHECK:           stablehlo.return %[[NEXT]], %[[STATE0]], %[[UPDATED]]
// CHECK:         }
// CHECK:         return %[[LB]], %[[UB]], %[[STEP]], %[[RESULTS]]#1, %[[RESULTS]]#2
// CHECK: enzymexla.temp_alloc "private" @[[LB_TEMP]] : memref<i32, 1>
// CHECK: enzymexla.temp_alloc "private" @[[UB_TEMP]] : memref<i32, 1>
// CHECK: enzymexla.temp_alloc "private" @[[STEP_TEMP]] : memref<i32, 1>

// CHECK-LABEL: llvm.func @lift_static_loop(
// CHECK-NOT:     scf.for
// CHECK-NOT:     enzymexla.get_global_temp
// CHECK:         %[[STATIC_ARG0:.*]] = "enzymexla.pointer2memref"
// CHECK:         %[[STATIC_ARG1:.*]] = "enzymexla.pointer2memref"
// CHECK:         enzymexla.xla_wrapper @[[STATIC_KERNEL:static_loop_body]]
// CHECK-SAME:      (%[[STATIC_ARG0]], %[[STATIC_ARG1]])
// CHECK-NOT:     enzymexla.get_global_temp
// CHECK:         llvm.return

// CHECK:       func.func private @[[STATIC_KERNEL]](
// CHECK-SAME:      %[[STATIC_BUFFER0:.*]]: tensor<4xf32>, %[[STATIC_BUFFER1:.*]]: tensor<4xf32>
// CHECK:         %[[START:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK:         %[[LIMIT:.*]] = stablehlo.constant dense<5> : tensor<i32>
// CHECK:         %[[STEP:.*]] = stablehlo.constant dense<2> : tensor<i32>
// CHECK:         %[[STATIC_RESULTS:.*]]:3 = stablehlo.while(%[[ITER:.*]] = %[[START]],
// CHECK:         stablehlo.compare LT, %[[ITER]], %[[LIMIT]], SIGNED
// CHECK:         stablehlo.add %[[ITER]], %[[STEP]]
// CHECK:         stablehlo.subtract
// CHECK:         return %[[STATIC_RESULTS]]#1, %[[STATIC_RESULTS]]#2

// CHECK-NOT: enzymexla.temp_alloc

// CHECK-LABEL: llvm.func @lift_without_noalias
// CHECK-NOT: scf.for
// CHECK: enzymexla.get_global_temp
// CHECK: enzymexla.xla_wrapper @[[UNANNOTATED_KERNEL:may_alias_loop_body]]
// CHECK: llvm.return
// CHECK: func.func private @[[UNANNOTATED_KERNEL]](
// CHECK: stablehlo.while
// CHECK: stablehlo.subtract

// -----

// Two host functions can use the same raised body. Each loop must have
// a separate bound allocation so one call cannot change another bound.
module {
  llvm.func @first_call_site(%limit: i32, %a: !llvm.ptr) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    scf.for %iv = %zero to %limit step %one : i32 {
      %buffer = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf32>
      enzymexla.xla_wrapper @shared_body(%buffer) : (memref<?xf32>) -> ()
    }
    llvm.return
  }
  llvm.func @second_call_site(%limit: i32, %a: !llvm.ptr) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    scf.for %iv = %zero to %limit step %one : i32 {
      %buffer = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf32>
      enzymexla.xla_wrapper @shared_body(%buffer) : (memref<?xf32>) -> ()
    }
    llvm.return
  }
  func.func private @shared_body(%a: tensor<?xf32>) -> tensor<?xf32> {
    %updated = stablehlo.add %a, %a : tensor<?xf32>
    return %updated : tensor<?xf32>
  }
}

// CHECK-LABEL: llvm.func @first_call_site(
// CHECK: %[[FIRST:.*]] = enzymexla.get_global_temp @[[$FIRST_TEMP:.*]] : memref<i32, 1>
// CHECK: enzymexla.memcpy %[[FIRST]],
// CHECK: enzymexla.xla_wrapper @{{.*}} (%[[FIRST]],
// CHECK-LABEL: llvm.func @second_call_site(
// CHECK: %[[SECOND:.*]] = enzymexla.get_global_temp @[[SECOND_TEMP:.*]] : memref<i32, 1>
// CHECK: enzymexla.memcpy %[[SECOND]],
// CHECK: enzymexla.xla_wrapper @{{.*}} (%[[SECOND]],
// CHECK-DAG: enzymexla.temp_alloc "private" @[[$FIRST_TEMP]] : memref<i32, 1>
// CHECK-DAG: enzymexla.temp_alloc "private" @[[SECOND_TEMP]] : memref<i32, 1>

// -----

// A new bound argument would change the position of argument attributes.
// Keep the loop unchanged until these attributes can be moved safely.
module {
  llvm.func @keep_argument_attributes(%limit: i32, %a: !llvm.ptr) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    scf.for %iv = %zero to %limit step %one : i32 {
      %buffer = "enzymexla.pointer2memref"(%a) : (!llvm.ptr) -> memref<?xf32>
      enzymexla.xla_wrapper @annotated_body(%buffer) : (memref<?xf32>) -> ()
    }
    llvm.return
  }
  func.func private @annotated_body(%a: tensor<?xf32> {test.keep})
      -> tensor<?xf32> {
    %updated = stablehlo.add %a, %a : tensor<?xf32>
    return %updated : tensor<?xf32>
  }
}

// CHECK-LABEL: llvm.func @keep_argument_attributes(
// CHECK: scf.for
// CHECK: enzymexla.xla_wrapper @annotated_body
// CHECK: func.func private @annotated_body(%{{.*}}: tensor<?xf32> {test.keep})
// CHECK-NOT: enzymexla.temp_alloc

// WhileLoopInfo recognizes the captured constants: [1, 5) with step 2 has
// two iterations. The second update must consume the first update's result.
// UNROLL-LABEL: llvm.func @lift_static_loop(
// UNROLL: enzymexla.xla_wrapper @[[STATIC_KERNEL:static_loop_body]]
// UNROLL: func.func private @[[STATIC_KERNEL]](
// UNROLL-SAME: %[[A:[^:]+]]: tensor<4xf32>, %[[B:[^:]+]]: tensor<4xf32>)
// UNROLL-NEXT: %[[FIRST:[^ ]+]] = stablehlo.subtract %[[A]], %[[B]] : tensor<4xf32>
// UNROLL-NEXT: %[[SECOND:[^ ]+]] = stablehlo.subtract %[[FIRST]], %[[B]] : tensor<4xf32>
// UNROLL-NEXT: return %[[SECOND]], %[[B]] : tensor<4xf32>, tensor<4xf32>
// UNROLL-NEXT: }
