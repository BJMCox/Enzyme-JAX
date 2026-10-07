// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

// Loops over a buffer laid out by runtime sizes, as a kernel's
// Reshape(op, coeffDim, NQ, NE) reads it at c + coeffDim * (q + NQ * e).

// The rotation of a loop that might run no times leaves its bound as
// max(n, 1), while its accesses stride by n: the iterations writing rows
// 2e and 2e + 1 of width n then seem to meet. Past a check that n is
// positive the bound is n, and they stay apart.
llvm.func @abort() attributes {noreturn}
func.func @rows(%x: memref<?xf64>, %y: memref<?xf64>, %n32: i32, %ne: index) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %n = arith.index_cast %n32 : i32 to index
  %n64 = arith.extui %n32 : i32 to i64
  %m = arith.maxsi %n64, %c1_i64 : i64
  %bound = arith.index_cast %m : i64 to index
  %bad = arith.cmpi sle, %n32, %c0_i32 : i32
  cf.cond_br %bad, ^fail, ^ok
^fail:
  llvm.call @abort() : () -> ()
  llvm.unreachable
^ok:
  affine.for %e = 0 to %ne {
    affine.for %i = 0 to %bound {
      %v = affine.load %x[%i + %e * symbol(%n)] : memref<?xf64>
      affine.store %v, %y[%i + (%e * 2) * symbol(%n)] : memref<?xf64>
      affine.store %v, %y[%i + (%e * 2 + 1) * symbol(%n)] : memref<?xf64>
    }
  }
  return
}

// CHECK:  func.func @rows(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: i32, %arg3: index) {
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:   %0 = arith.index_cast %arg2 : i32 to index
// CHECK-NEXT:   %1 = arith.extui %arg2 : i32 to i64
// CHECK-NEXT:   %2 = arith.maxsi %1, %c1_i64 : i64
// CHECK-NEXT:   %3 = arith.index_cast %2 : i64 to index
// CHECK-NEXT:   %4 = arith.cmpi sle, %arg2, %c0_i32 : i32
// CHECK-NEXT:   cf.cond_br %4, ^bb1, ^bb2
// CHECK-NEXT: ^bb1:  // pred: ^bb0
// CHECK-NEXT:   llvm.call @abort() : () -> ()
// CHECK-NEXT:   llvm.unreachable
// CHECK-NEXT: ^bb2:  // pred: ^bb0
// CHECK-NEXT:   affine.parallel (%arg4, %arg5) = (0, 0) to (symbol(%arg3), symbol(%3)) {
// CHECK-NEXT:     %5 = affine.load %arg0[%arg5 + %arg4 * symbol(%0)] : memref<?xf64>
// CHECK-NEXT:     affine.store %5, %arg1[%arg5 + (%arg4 * symbol(%0)) * 2] : memref<?xf64>
// CHECK-NEXT:     affine.store %5, %arg1[%arg5 + (%arg4 * 2 + 1) * symbol(%0)] : memref<?xf64>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// Without the check, n may be 0 with the loop running once at i = 0: the
// bound says nothing about n, and the nest stays serial.
func.func @unchecked(%x: memref<?xf64>, %y: memref<?xf64>, %n32: i32, %ne: index) {
  %c1_i64 = arith.constant 1 : i64
  %n = arith.index_cast %n32 : i32 to index
  %n64 = arith.extui %n32 : i32 to i64
  %m = arith.maxsi %n64, %c1_i64 : i64
  %bound = arith.index_cast %m : i64 to index
  affine.for %e = 0 to %ne {
    affine.for %i = 0 to %bound {
      %v = affine.load %x[%i + %e * symbol(%n)] : memref<?xf64>
      affine.store %v, %y[%i + (%e * 2) * symbol(%n)] : memref<?xf64>
      affine.store %v, %y[%i + (%e * 2 + 1) * symbol(%n)] : memref<?xf64>
    }
  }
  return
}

// CHECK:  func.func @unchecked(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: i32, %arg3: index) {
// CHECK-NEXT:   %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:   %0 = arith.index_cast %arg2 : i32 to index
// CHECK-NEXT:   %1 = arith.extui %arg2 : i32 to i64
// CHECK-NEXT:   %2 = arith.maxsi %1, %c1_i64 : i64
// CHECK-NEXT:   %3 = arith.index_cast %2 : i64 to index
// CHECK-NEXT:   affine.for %arg4 = 0 to %arg3 {
// CHECK-NEXT:     affine.for %arg5 = 0 to %3 {
// CHECK-NEXT:       %4 = affine.load %arg0[%arg5 + %arg4 * symbol(%0)] : memref<?xf64>
// CHECK-NEXT:       affine.store %4, %arg1[%arg5 + (%arg4 * symbol(%0)) * 2] : memref<?xf64>
// CHECK-NEXT:       affine.store %4, %arg1[%arg5 + (%arg4 * 2 + 1) * symbol(%0)] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// The count is read as an index two ways, cast directly and through a zero
// extension; a check that it is positive makes them one value, and the
// width the nest strides by is the bound of the loop it is read in.
func.func @index_forms(%n32: i32, %x: memref<?xf64>, %y: memref<?xf64>) {
  %c0_i32 = arith.constant 0 : i32
  %pos = arith.cmpi sgt, %n32, %c0_i32 : i32
  %n = arith.index_cast %n32 : i32 to index
  %n64 = arith.extui %n32 : i32 to i64
  %w = arith.index_cast %n64 : i64 to index
  scf.if %pos {
    affine.for %e = 0 to %n {
      affine.for %i = 0 to %n {
        %v = affine.load %x[%i + %e * symbol(%w)] : memref<?xf64>
        affine.store %v, %y[%i + (%e * 2) * symbol(%w)] : memref<?xf64>
        affine.store %v, %y[%i + (%e * 2 + 1) * symbol(%w)] : memref<?xf64>
      }
    }
  }
  return
}

// CHECK:  func.func @index_forms(%arg0: i32, %arg1: memref<?xf64>, %arg2: memref<?xf64>) {
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %0 = arith.cmpi sgt, %arg0, %c0_i32 : i32
// CHECK-NEXT:   %1 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:   %2 = arith.extui %arg0 : i32 to i64
// CHECK-NEXT:   %3 = arith.index_cast %2 : i64 to index
// CHECK-NEXT:   scf.if %0 {
// CHECK-NEXT:     affine.parallel (%arg3, %arg4) = (0, 0) to (symbol(%1), symbol(%1)) {
// CHECK-NEXT:       %4 = affine.load %arg1[%arg4 + %arg3 * symbol(%3)] : memref<?xf64>
// CHECK-NEXT:       affine.store %4, %arg2[%arg4 + (%arg3 * symbol(%3)) * 2] : memref<?xf64>
// CHECK-NEXT:       affine.store %4, %arg2[%arg4 + (%arg3 * 2 + 1) * symbol(%3)] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// NQ = Q1D * Q1D is at least 1 past a check that Q1D is not zero: the loop
// over q < NQ stays in its row of width NQ.
func.func @square(%q1d: i32, %x: memref<?xf64>, %y: memref<?xf64>, %ne: index) {
  %c0_i32 = arith.constant 0 : i32
  %nz = arith.cmpi ne, %q1d, %c0_i32 : i32
  %nq32 = arith.muli %q1d, %q1d overflow<nsw> : i32
  %nq = arith.index_cast %nq32 : i32 to index
  scf.if %nz {
    affine.for %e = 0 to %ne {
      affine.for %q = 0 to %nq {
        %v = affine.load %x[%q + %e * symbol(%nq)] : memref<?xf64>
        affine.store %v, %y[%q + (%e * 2) * symbol(%nq)] : memref<?xf64>
        affine.store %v, %y[%q + (%e * 2 + 1) * symbol(%nq)] : memref<?xf64>
      }
    }
  }
  return
}

// CHECK:  func.func @square(%arg0: i32, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: index) {
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %0 = arith.cmpi ne, %arg0, %c0_i32 : i32
// CHECK-NEXT:   %1 = arith.muli %arg0, %arg0 overflow<nsw> : i32
// CHECK-NEXT:   %2 = arith.index_cast %1 : i32 to index
// CHECK-NEXT:   scf.if %0 {
// CHECK-NEXT:     affine.parallel (%arg4, %arg5) = (0, 0) to (symbol(%arg3), symbol(%2)) {
// CHECK-NEXT:       %3 = affine.load %arg1[%arg5 + %arg4 * symbol(%2)] : memref<?xf64>
// CHECK-NEXT:       affine.store %3, %arg2[%arg5 + (%arg4 * symbol(%2)) * 2] : memref<?xf64>
// CHECK-NEXT:       affine.store %3, %arg2[%arg5 + (%arg4 * 2 + 1) * symbol(%2)] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// y(c, q, e) = w(q) * x(c, q, e) over the three runtime sizes, each loop
// rotated to run at least once: the flat index c + cd * (q + nq * e) splits
// into (e, q, c), each below its size, and the nest is parallel throughout.
func.func @three_levels(%w: memref<?xf64>, %x: memref<?xf64>, %y: memref<?xf64>, %nq32: i32, %cd32: i32, %ne32: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %nq = arith.index_cast %nq32 : i32 to index
  %cd = arith.index_cast %cd32 : i32 to index
  %nq64 = arith.extui %nq32 : i32 to i64
  %bq64 = arith.maxsi %nq64, %c1_i64 : i64
  %bq = arith.index_cast %bq64 : i64 to index
  %cd64 = arith.extui %cd32 : i32 to i64
  %bc64 = arith.maxsi %cd64, %c1_i64 : i64
  %bc = arith.index_cast %bc64 : i64 to index
  %ne64 = arith.extui %ne32 : i32 to i64
  %be64 = arith.maxsi %ne64, %c1_i64 : i64
  %be = arith.index_cast %be64 : i64 to index
  %pe = arith.cmpi sgt, %ne32, %c0_i32 : i32
  scf.if %pe {
    %pq = arith.cmpi sgt, %nq32, %c0_i32 : i32
    %pc = arith.cmpi sgt, %cd32, %c0_i32 : i32
    %p = arith.andi %pq, %pc : i1
    scf.if %p {
      affine.for %e = 0 to %be {
        affine.for %q = 0 to %bq {
          affine.parallel (%c) = (0) to (symbol(%bc)) {
            %wv = affine.load %w[%q] : memref<?xf64>
            %xv = affine.load %x[%c + (%q + %e * symbol(%nq)) * symbol(%cd)] : memref<?xf64>
            %m = arith.mulf %wv, %xv : f64
            affine.store %m, %y[%c + (%q + %e * symbol(%nq)) * symbol(%cd)] : memref<?xf64>
          }
        }
      }
    }
  }
  return
}

// CHECK:  func.func @three_levels(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: i32, %arg4: i32, %arg5: i32) {
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:   %0 = arith.index_cast %arg3 : i32 to index
// CHECK-NEXT:   %1 = arith.index_cast %arg4 : i32 to index
// CHECK-NEXT:   %2 = arith.extui %arg3 : i32 to i64
// CHECK-NEXT:   %3 = arith.maxsi %2, %c1_i64 : i64
// CHECK-NEXT:   %4 = arith.index_cast %3 : i64 to index
// CHECK-NEXT:   %5 = arith.extui %arg4 : i32 to i64
// CHECK-NEXT:   %6 = arith.maxsi %5, %c1_i64 : i64
// CHECK-NEXT:   %7 = arith.index_cast %6 : i64 to index
// CHECK-NEXT:   %8 = arith.extui %arg5 : i32 to i64
// CHECK-NEXT:   %9 = arith.maxsi %8, %c1_i64 : i64
// CHECK-NEXT:   %10 = arith.index_cast %9 : i64 to index
// CHECK-NEXT:   %11 = arith.cmpi sgt, %arg5, %c0_i32 : i32
// CHECK-NEXT:   scf.if %11 {
// CHECK-NEXT:     %12 = arith.cmpi sgt, %arg3, %c0_i32 : i32
// CHECK-NEXT:     %13 = arith.cmpi sgt, %arg4, %c0_i32 : i32
// CHECK-NEXT:     %14 = arith.andi %12, %13 : i1
// CHECK-NEXT:     scf.if %14 {
// CHECK-NEXT:       affine.parallel (%arg6, %arg7, %arg8) = (0, 0, 0) to (symbol(%10), symbol(%4), symbol(%7)) {
// CHECK-NEXT:         %15 = affine.load %arg0[%arg7] : memref<?xf64>
// CHECK-NEXT:         %16 = affine.load %arg1[%arg8 + (%arg7 + %arg6 * symbol(%0)) * symbol(%1)] : memref<?xf64>
// CHECK-NEXT:         %17 = arith.mulf %15, %16 : f64
// CHECK-NEXT:         affine.store %17, %arg2[%arg8 + (%arg7 + %arg6 * symbol(%0)) * symbol(%1)] : memref<?xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// A layout whose row count a flag picks, y(q, k, e) at q + nq * (k + sel * e)
// with sel 3 or 4: rows k = 0, 1, 2 of block e are below sel, and the blocks
// nq * sel apart.
func.func @chosen_rows(%x: memref<?xf64>, %y: memref<?xf64>, %q1d: i32, %cd: i32, %ne: index) {
  %c0_i32 = arith.constant 0 : i32
  %c4_i32 = arith.constant 4 : i32
  %c1_i64 = arith.constant 1 : i64
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %sym = arith.cmpi ne, %cd, %c4_i32 : i32
  %sel = arith.select %sym, %c3, %c4 : index
  %nq32 = arith.muli %q1d, %q1d overflow<nsw> : i32
  %nq = arith.index_cast %nq32 : i32 to index
  %nq64 = arith.extui %nq32 : i32 to i64
  %bq64 = arith.maxsi %nq64, %c1_i64 : i64
  %bq = arith.index_cast %bq64 : i64 to index
  %nz = arith.cmpi ne, %q1d, %c0_i32 : i32
  scf.if %nz {
    affine.for %e = 0 to %ne {
      affine.for %q = 0 to %bq {
        %v = affine.load %x[%q + %e * symbol(%nq)] : memref<?xf64>
        affine.store %v, %y[%q + (%e * symbol(%sel)) * symbol(%nq)] : memref<?xf64>
        affine.store %v, %y[%q + (%e * symbol(%sel) + 1) * symbol(%nq)] : memref<?xf64>
        affine.store %v, %y[%q + (%e * symbol(%sel) + 2) * symbol(%nq)] : memref<?xf64>
      }
    }
  }
  return
}

// CHECK:  func.func @chosen_rows(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: i32, %arg3: i32, %arg4: index) {
// CHECK-NEXT:   %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:   %c4_i32 = arith.constant 4 : i32
// CHECK-NEXT:   %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:   %c3 = arith.constant 3 : index
// CHECK-NEXT:   %c4 = arith.constant 4 : index
// CHECK-NEXT:   %0 = arith.cmpi ne, %arg3, %c4_i32 : i32
// CHECK-NEXT:   %1 = arith.select %0, %c3, %c4 : index
// CHECK-NEXT:   %2 = arith.muli %arg2, %arg2 overflow<nsw> : i32
// CHECK-NEXT:   %3 = arith.index_cast %2 : i32 to index
// CHECK-NEXT:   %4 = arith.extui %2 : i32 to i64
// CHECK-NEXT:   %5 = arith.maxsi %4, %c1_i64 : i64
// CHECK-NEXT:   %6 = arith.index_cast %5 : i64 to index
// CHECK-NEXT:   %7 = arith.cmpi ne, %arg2, %c0_i32 : i32
// CHECK-NEXT:   scf.if %7 {
// CHECK-NEXT:     affine.parallel (%arg5, %arg6) = (0, 0) to (symbol(%arg4), symbol(%6)) {
// CHECK-NEXT:       %8 = affine.load %arg0[%arg6 + %arg5 * symbol(%3)] : memref<?xf64>
// CHECK-NEXT:       affine.store %8, %arg1[%arg6 + (%arg5 * symbol(%1)) * symbol(%3)] : memref<?xf64>
// CHECK-NEXT:       affine.store %8, %arg1[%arg6 + (%arg5 * symbol(%1) + 1) * symbol(%3)] : memref<?xf64>
// CHECK-NEXT:       affine.store %8, %arg1[%arg6 + (%arg5 * symbol(%1) + 2) * symbol(%3)] : memref<?xf64>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
