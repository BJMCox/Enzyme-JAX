// RUN: enzymexlamlir-opt %s --lower-small-tensor-values --outline-small-gpu-program | FileCheck %s

func.func @main(%bits: tensor<2x3xi1>, %words: tensor<2xui32>) -> (tensor<2x3xi1>, tensor<2xui32>) {
  return %bits, %words : tensor<2x3xi1>, tensor<2xui32>
}

// CHECK-LABEL: func.func private @main$small_value
// CHECK-SAME: vector<6xi1>
// CHECK-SAME: vector<2xi32>
// CHECK-LABEL: func.func private @main$small_kernel
// CHECK: llvm.load %arg0 <alignment = 1> : !llvm.ptr<1> -> vector<6xi8>
// CHECK: arith.trunci {{.*}} : vector<6xi8> to vector<6xi1>
// CHECK: llvm.load %arg1 <alignment = 4> : !llvm.ptr<1> -> vector<2xi32>
// CHECK: arith.extui {{.*}} : vector<6xi1> to vector<6xi8>
// CHECK: llvm.store {{.*}}, %arg2 <alignment = 1> : vector<6xi8>, !llvm.ptr<1>
// CHECK: llvm.store {{.*}}, %arg3 <alignment = 4> : vector<2xi32>, !llvm.ptr<1>
// CHECK-LABEL: func.func @main(
// CHECK-SAME: tensor<2x3xi1>
// CHECK-SAME: tensor<2xui32>
// CHECK: enzymexla.kernel_call @main$small_kernel
// CHECK-SAME: operand_layouts = [dense<[1, 0]>
// CHECK-SAME: output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 2
// CHECK-SAME: output_tuple_indices = [1], operand_index = 3
