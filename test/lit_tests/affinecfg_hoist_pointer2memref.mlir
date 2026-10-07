// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// The same view of the output in both arms of a branch is two memref values
// of one buffer to the dependence analysis, which then takes a loop writing
// through one as aliasing two buffers. The view is taken where the pointer
// is, once, and the loop is parallel.
module {
  func.func @arms(%p: !llvm.ptr, %x: memref<?xf64>, %n: index, %flag: i1) {
    affine.for %i = 0 to %n {
      %v = affine.load %x[%i] : memref<?xf64>
      scf.if %flag {
        %m = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xf64>
        affine.store %v, %m[%i] : memref<?xf64>
      } else {
        %m = "enzymexla.pointer2memref"(%p) : (!llvm.ptr) -> memref<?xf64>
        %w = arith.negf %v : f64
        affine.store %w, %m[%i] : memref<?xf64>
      }
    }
    return
  }
}

// CHECK:  func.func @arms(%arg0: !llvm.ptr, %arg1: memref<?xf64>, %arg2: index, %arg3: i1) {
// CHECK-NEXT:    %0 = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<?xf64>
// CHECK-NEXT:    affine.parallel (%arg4) = (0) to (symbol(%arg2)) {
// CHECK-NEXT:      %1 = affine.load %arg1[%arg4] : memref<?xf64>
// CHECK-NEXT:      %2 = scf.if %arg3 -> (f64) {
// CHECK-NEXT:        scf.yield %1 : f64
// CHECK-NEXT:      } else {
// CHECK-NEXT:        %3 = arith.negf %1 : f64
// CHECK-NEXT:        scf.yield %3 : f64
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.store %2, %0[%arg4] : memref<?xf64>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
