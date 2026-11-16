//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/SymbolTable.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::enzyme {
#define GEN_PASS_DEF_OUTLINESMALLGPUPROGRAMPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace mlir::enzyme

using namespace mlir;

namespace {

static std::string unusedName(SymbolTable &symbols, StringRef base) {
  std::string name = base.str();
  for (unsigned i = 1; symbols.lookup(name); ++i)
    name = (base + Twine(i)).str();
  return name;
}

struct OutlineSmallGPUProgramPass
    : mlir::enzyme::impl::OutlineSmallGPUProgramPassBase<
          OutlineSmallGPUProgramPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto program = module.lookupSymbol<func::FuncOp>(entryFunction.getValue());
    auto saved =
        program
            ? program->getAttrOfType<TypeAttr>("enzymexla.small_program_type")
            : TypeAttr();
    auto signature =
        saved ? dyn_cast<FunctionType>(saved.getValue()) : FunctionType();
    if (!program || !signature || program.isExternal()) {
      module.emitError("requires an entry from lower-small-tensor-values");
      return signalPassFailure();
    }
    FunctionType vectorSignature = program.getFunctionType();
    if (signature.getNumInputs() != vectorSignature.getNumInputs() ||
        signature.getNumResults() != vectorSignature.getNumResults()) {
      program.emitError("tensor and vector entry signatures disagree");
      return signalPassFailure();
    }
    SmallVector<Type> tensorTypes(signature.getInputs());
    llvm::append_range(tensorTypes, signature.getResults());
    SmallVector<Type> vectorTypes(vectorSignature.getInputs());
    llvm::append_range(vectorTypes, vectorSignature.getResults());
    if (tensorTypes.size() > 512) {
      program.emitError("small GPU profile supports at most 512 buffers");
      return signalPassFailure();
    }
    uint64_t payload = 0;
    uint64_t scratchBytes = 1;
    for (auto [tensor, vector] : llvm::zip_equal(tensorTypes, vectorTypes)) {
      auto shaped = dyn_cast<RankedTensorType>(tensor);
      auto flat = dyn_cast<VectorType>(vector);
      if (!shaped || !shaped.hasStaticShape() || shaped.getEncoding() ||
          !flat || flat.getRank() != 1 || flat.isScalable() ||
          shaped.getNumElements() != flat.getNumElements() ||
          shaped.getElementTypeBitWidth() != flat.getElementTypeBitWidth()) {
        program.emitError("entry buffer does not match its vector value");
        return signalPassFailure();
      }
      Type element = shaped.getElementType();
      if (auto integer = dyn_cast<IntegerType>(element))
        element = IntegerType::get(module.getContext(), integer.getWidth());
      if (element != flat.getElementType()) {
        program.emitError(
            "entry buffer element type does not match its vector value");
        return signalPassFailure();
      }
      uint64_t bytes =
          shaped.getNumElements() * ((shaped.getElementTypeBitWidth() + 7) / 8);
      if (bytes > 65536 || payload > 65536 - bytes) {
        program.emitError("small GPU entry buffer payload exceeds 64 KiB");
        return signalPassFailure();
      }
      payload += bytes;
      scratchBytes = std::max(scratchBytes, bytes);
    }

    SymbolTable symbols(module);
    std::string entryName = entryFunction.getValue();
    std::string programName = unusedName(symbols, entryName + "$small_value");
    if (failed(symbols.rename(program, programName)))
      return signalPassFailure();
    program.setPrivate();
    program->removeAttr("enzymexla.small_program_type");
    program->removeAttr("arg_attrs");
    program->removeAttr("res_attrs");

    OpBuilder builder(module.getContext());
    builder.setInsertionPointAfter(program);
    Location loc = program.getLoc();
    Type pointer = LLVM::LLVMPointerType::get(module.getContext(), 1);
    SmallVector<Type> pointers(tensorTypes.size(), pointer);
    std::string kernelName = unusedName(symbols, entryName + "$small_kernel");
    auto kernel = func::FuncOp::create(builder, loc, kernelName,
                                       builder.getFunctionType(pointers, {}));
    kernel.setPrivate();
    symbols.insert(kernel);
    builder.setInsertionPointToStart(kernel.addEntryBlock());

    auto constant = [&](uint64_t value) -> Value {
      return arith::ConstantIntOp::create(builder, loc, builder.getI64Type(),
                                          value);
    };
    Value scratch = LLVM::AllocaOp::create(
        builder, loc, LLVM::LLVMPointerType::get(module.getContext()),
        builder.getI8Type(), constant(scratchBytes), 8);
    auto aligned = [&](Value pointer, unsigned alignment) -> Value {
      Value address =
          LLVM::PtrToIntOp::create(builder, loc, builder.getI64Type(), pointer);
      Value low =
          arith::AndIOp::create(builder, loc, address, constant(alignment - 1));
      return arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, low,
                                   constant(0));
    };
    auto load = [&](VectorType type, Value pointer) -> Value {
      unsigned alignment = type.getElementTypeBitWidth() / 8;
      if (alignment == 1)
        return LLVM::LoadOp::create(builder, loc, type, pointer, 1);
      Value condition = aligned(pointer, alignment);
      auto bytes = builder.getI64IntegerAttr(type.getNumElements() * alignment);
      auto result = scf::IfOp::create(
          builder, loc, condition,
          [&](OpBuilder &nested, Location loc) {
            Value value =
                LLVM::LoadOp::create(nested, loc, type, pointer, alignment);
            scf::YieldOp::create(nested, loc, value);
          },
          [&](OpBuilder &nested, Location loc) {
            // Keep complete element bits when a foreign buffer lacks natural
            // alignment. Volatile byte copies preserve this memory boundary.
            LLVM::MemcpyInlineOp::create(nested, loc, scratch, pointer, bytes,
                                         true);
            Value value =
                LLVM::LoadOp::create(nested, loc, type, scratch, alignment);
            scf::YieldOp::create(nested, loc, value);
          });
      return result.getResult(0);
    };
    auto store = [&](Value value, Value pointer) {
      auto type = cast<VectorType>(value.getType());
      unsigned alignment = type.getElementTypeBitWidth() / 8;
      if (alignment == 1) {
        LLVM::StoreOp::create(builder, loc, value, pointer, 1);
        return;
      }
      Value condition = aligned(pointer, alignment);
      auto bytes = builder.getI64IntegerAttr(type.getNumElements() * alignment);
      scf::IfOp::create(
          builder, loc, condition,
          [&](OpBuilder &nested, Location loc) {
            LLVM::StoreOp::create(nested, loc, value, pointer, alignment);
            scf::YieldOp::create(nested, loc);
          },
          [&](OpBuilder &nested, Location loc) {
            LLVM::StoreOp::create(nested, loc, value, scratch, alignment);
            LLVM::MemcpyInlineOp::create(nested, loc, pointer, scratch, bytes,
                                         true);
            scf::YieldOp::create(nested, loc);
          });
    };

    SmallVector<Value> arguments;
    unsigned inputs = vectorSignature.getNumInputs();
    for (unsigned i = 0; i < inputs; ++i) {
      auto vector = cast<VectorType>(vectorTypes[i]);
      bool boolean = vector.getElementType().isInteger(1);
      auto storage =
          boolean ? VectorType::get(vector.getShape(), builder.getI8Type())
                  : vector;
      Value value = load(storage, kernel.getArgument(i));
      if (boolean)
        value = arith::TruncIOp::create(builder, loc, vector, value);
      arguments.push_back(value);
    }
    auto call = func::CallOp::create(builder, loc, program, arguments);
    for (auto [i, result] : llvm::enumerate(call.getResults())) {
      auto vector = cast<VectorType>(result.getType());
      Value value = result;
      if (vector.getElementType().isInteger(1))
        value = arith::ExtUIOp::create(
            builder, loc,
            VectorType::get(vector.getShape(), builder.getI8Type()), value);
      store(value, kernel.getArgument(inputs + i));
    }
    func::ReturnOp::create(builder, loc);

    builder.setInsertionPointAfter(kernel);
    auto entry = func::FuncOp::create(builder, loc, entryFunction, signature);
    builder.setInsertionPointToStart(entry.addEntryBlock());
    SmallVector<Value> buffers(entry.getArguments());
    SmallVector<Attribute> aliases;
    for (auto [i, type] : llvm::enumerate(signature.getResults())) {
      auto tensor = cast<RankedTensorType>(type);
      Attribute zero;
      if (auto floating = dyn_cast<FloatType>(tensor.getElementType()))
        zero = builder.getFloatAttr(floating, 0.0);
      else
        zero = builder.getIntegerAttr(tensor.getElementType(), 0);
      buffers.push_back(stablehlo::ConstantOp::create(
          builder, loc, DenseElementsAttr::get(tensor, zero)));
      SmallVector<int64_t> resultIndex;
      if (signature.getNumResults() != 1)
        resultIndex.push_back(i);
      aliases.push_back(stablehlo::OutputOperandAliasAttr::get(
          module.getContext(), resultIndex, inputs + i, {}));
    }
    SmallVector<Attribute> layouts;
    for (Type type : tensorTypes) {
      auto tensor = cast<RankedTensorType>(type);
      SmallVector<int64_t> dimensions;
      for (int64_t i = tensor.getRank(); i > 0; --i)
        dimensions.push_back(i - 1);
      layouts.push_back(DenseIntElementsAttr::get(
          RankedTensorType::get({tensor.getRank()}, builder.getIndexType()),
          ArrayRef<int64_t>(dimensions)));
    }
    auto scalarType = RankedTensorType::get({}, builder.getI64Type());
    Value one = stablehlo::ConstantOp::create(
        builder, loc,
        DenseElementsAttr::get(scalarType, builder.getI64IntegerAttr(1)));
    Value zero = stablehlo::ConstantOp::create(
        builder, loc,
        DenseElementsAttr::get(scalarType, builder.getI64IntegerAttr(0)));
    auto launch = enzymexla::KernelCallOp::create(
        builder, loc, signature.getResults(),
        SymbolRefAttr::get(module.getContext(), kernelName), one, one, one, one,
        one, one, zero, Value(), Value(), Value(), buffers, "",
        builder.getArrayAttr(layouts), ArrayAttr(), ArrayAttr(),
        builder.getArrayAttr(aliases), builder.getUnitAttr());
    func::ReturnOp::create(builder, loc, launch.getResults());
  }
};

} // namespace
