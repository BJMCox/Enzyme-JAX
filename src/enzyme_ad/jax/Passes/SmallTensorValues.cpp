//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/enzyme_ad/jax/Passes/Passes.h"
namespace mlir::enzyme {
#define GEN_PASS_DEF_LOWERSMALLTENSORVALUESPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace mlir::enzyme
using namespace mlir;
namespace {

static VectorType flatType(RankedTensorType type) {
  Type element = type.getElementType();
  if (auto integer = dyn_cast<IntegerType>(element))
    element = IntegerType::get(type.getContext(), integer.getWidth());
  return VectorType::get({type.getNumElements()}, element);
}

struct Values : ConversionPattern {
  Values(TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, context) {
    // Payloads can contain tensor accesses. Each rewrite removes its parent
    // operation and clones its payload once, so recursive legalization is
    // finite.
    setHasBoundedRewriteRecursion();
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto index = [&](int64_t value) -> Value {
      return arith::ConstantIndexOp::create(rewriter, loc, value);
    };
    auto linearIndex = [&](ArrayRef<Value> indices,
                           ArrayRef<int64_t> shape) -> Value {
      Value value = index(0);
      for (auto [i, size] : llvm::zip_equal(indices, shape)) {
        value = arith::MulIOp::create(rewriter, loc, value, index(size));
        value = arith::AddIOp::create(rewriter, loc, value, i);
      }
      return value;
    };
    auto extract = [&](Value vector, Value position) -> Value {
      return vector::ExtractOp::create(rewriter, loc, vector, position);
    };
    auto insert = [&](Value scalar, Value vector, Value position) -> Value {
      return vector::InsertOp::create(rewriter, loc, scalar, vector, position);
    };
    auto scalarRegion = [&](Block &block, ArrayRef<Value> args,
                            ArrayRef<Value> loopIndices) -> SmallVector<Value> {
      IRMapping mapping;
      for (auto [arg, value] : llvm::zip_equal(block.getArguments(), args))
        mapping.map(arg, value);
      for (Operation &nested : block.without_terminator()) {
        if (auto dim = dyn_cast<linalg::IndexOp>(nested)) {
          mapping.map(dim.getResult(), loopIndices[dim.getDim()]);
        } else {
          rewriter.clone(nested, mapping);
        }
      }
      SmallVector<Value> values;
      for (Value value : block.getTerminator()->getOperands())
        values.push_back(mapping.lookupOrDefault(value));
      return values;
    };
    if (isa<UnrealizedConversionCastOp>(op)) {
      SmallVector<Type> types;
      if (failed(
              getTypeConverter()->convertTypes(op->getResultTypes(), types)) ||
          TypeRange(operands) != TypeRange(types))
        return failure();
      rewriter.replaceOp(op, operands);
      return success();
    }
    if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      auto tensor = dyn_cast<RankedTensorType>(constant.getType());
      if (!tensor || !tensor.hasStaticShape())
        return failure();
      auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
      if (!dense)
        return failure();
      auto vector = flatType(tensor);
      if (dense.getElementType() != vector.getElementType())
        dense = dense.bitcast(vector.getElementType());
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, dense.reshape(vector));
      return success();
    }
    if (isa<arith::SelectOp>(op) &&
        isa<RankedTensorType>(op->getResult(0).getType())) {
      rewriter.replaceOpWithNewOp<arith::SelectOp>(op, operands[0], operands[1],
                                                   operands[2]);
      return success();
    }
    if (auto empty = dyn_cast<tensor::EmptyOp>(op)) {
      auto tensor = empty.getType();
      if (!tensor.hasStaticShape())
        return failure();
      rewriter.replaceOpWithNewOp<ub::PoisonOp>(op, flatType(tensor));
      return success();
    }
    if (isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp, tensor::CastOp>(
            op)) {
      if (operands[0].getType() !=
          getTypeConverter()->convertType(op->getResult(0).getType()))
        return failure();
      rewriter.replaceOp(op, operands[0]);
      return success();
    }
    if (auto get = dyn_cast<tensor::ExtractOp>(op)) {
      Value position = linearIndex(operands.drop_front(),
                                   get.getTensor().getType().getShape());
      rewriter.replaceOp(op, extract(operands[0], position));
      return success();
    }
    if (auto put = dyn_cast<tensor::InsertOp>(op)) {
      Value position = linearIndex(operands.drop_front(2),
                                   put.getDest().getType().getShape());
      rewriter.replaceOp(op, insert(operands[0], operands[1], position));
      return success();
    }
    if (auto from = dyn_cast<tensor::FromElementsOp>(op)) {
      rewriter.replaceOpWithNewOp<vector::FromElementsOp>(
          op, flatType(from.getType()), operands);
      return success();
    }
    if (auto fill = dyn_cast<linalg::FillOp>(op)) {
      auto type = getTypeConverter()->convertType(op->getResult(0).getType());
      rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, type, operands[0]);
      return success();
    }
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      if (!linalgOp.hasPureTensorSemantics())
        return failure();
      if (!llvm::all_of(op->getOperandTypes(),
                        [](Type type) { return isa<RankedTensorType>(type); }))
        return failure();
      SmallVector<int64_t> bounds = linalgOp.getStaticLoopRanges();
      if (llvm::any_of(bounds, [](int64_t bound) { return bound < 0; }))
        return failure();
      auto maps = linalgOp.getIndexingMapsArray();
      for (AffineMap map : maps)
        if (llvm::any_of(map.getResults(), [](AffineExpr expr) {
              return !isa<AffineDimExpr, AffineConstantExpr>(expr);
            }))
          return failure();
      unsigned inputCount = linalgOp.getNumDpsInputs();
      SmallVector<Value> outputs(operands.drop_front(inputCount));
      SmallVector<Value> indices;
      auto mappedIndex = [&](unsigned i) -> Value {
        SmallVector<Value> coords;
        for (AffineExpr expr : maps[i].getResults()) {
          if (auto dim = dyn_cast<AffineDimExpr>(expr))
            coords.push_back(indices[dim.getPosition()]);
          else
            coords.push_back(index(cast<AffineConstantExpr>(expr).getValue()));
        }
        return linearIndex(
            coords,
            cast<RankedTensorType>(op->getOperand(i).getType()).getShape());
      };
      std::function<SmallVector<Value>(unsigned, ValueRange)> loop;
      loop = [&](unsigned depth, ValueRange carried) -> SmallVector<Value> {
        if (depth == bounds.size()) {
          SmallVector<Value> args;
          for (unsigned i = 0; i < operands.size(); ++i)
            args.push_back(
                extract(i < inputCount ? operands[i] : carried[i - inputCount],
                        mappedIndex(i)));
          auto yielded = scalarRegion(op->getRegion(0).front(), args, indices);
          SmallVector<Value> result;
          for (auto [i, value] : llvm::enumerate(yielded))
            result.push_back(
                insert(value, carried[i], mappedIndex(inputCount + i)));
          return result;
        }
        auto forOp = scf::ForOp::create(
            rewriter, loc, index(0), index(bounds[depth]), index(1), carried,
            [&](OpBuilder &, Location, Value iv, ValueRange iterArgs) {
              indices.push_back(iv);
              auto next = loop(depth + 1, iterArgs);
              indices.pop_back();
              scf::YieldOp::create(rewriter, loc, next);
            });
        return SmallVector<Value>(forOp.getResults());
      };
      rewriter.replaceOp(op, loop(0, outputs));
      return success();
    }
    if (isa<tensor::ExtractSliceOp, tensor::InsertSliceOp>(op)) {
      bool extracting = isa<tensor::ExtractSliceOp>(op);
      auto offsets = op->getAttrOfType<DenseI64ArrayAttr>("static_offsets");
      auto sizes = op->getAttrOfType<DenseI64ArrayAttr>("static_sizes");
      auto strides = op->getAttrOfType<DenseI64ArrayAttr>("static_strides");
      if (!offsets || !sizes || !strides ||
          llvm::any_of(sizes.asArrayRef(), ShapedType::isDynamic))
        return failure();
      unsigned dynamic = extracting ? 1 : 2;
      SmallVector<Value> offsetValues, strideValues;
      for (int64_t value : offsets.asArrayRef())
        offsetValues.push_back(
            ShapedType::isDynamic(value) ? operands[dynamic++] : index(value));
      for (int64_t value : strides.asArrayRef())
        strideValues.push_back(
            ShapedType::isDynamic(value) ? operands[dynamic++] : index(value));
      auto sourceType = cast<RankedTensorType>(op->getOperand(0).getType());
      auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
      Value output =
          extracting ? ub::PoisonOp::create(rewriter, loc, flatType(resultType))
                           .getResult()
                     : operands[1];
      int64_t count = 1;
      for (int64_t size : sizes.asArrayRef())
        count *= size;
      auto forOp = scf::ForOp::create(
          rewriter, loc, index(0), index(count), index(1), ValueRange{output},
          [&](OpBuilder &, Location, Value iv, ValueRange iterArgs) {
            Value remaining = iv;
            SmallVector<Value> coords(sizes.size());
            for (int i = sizes.size() - 1; i >= 0; --i) {
              Value coord = arith::RemUIOp::create(rewriter, loc, remaining,
                                                   index(sizes[i]));
              remaining = arith::DivUIOp::create(rewriter, loc, remaining,
                                                 index(sizes[i]));
              coord =
                  arith::MulIOp::create(rewriter, loc, coord, strideValues[i]);
              coords[i] =
                  arith::AddIOp::create(rewriter, loc, coord, offsetValues[i]);
            }
            Value mapped =
                linearIndex(coords, extracting ? sourceType.getShape()
                                               : resultType.getShape());
            Value next =
                extracting
                    ? insert(extract(operands[0], mapped), iterArgs[0], iv)
                    : insert(extract(operands[0], iv), iterArgs[0], mapped);
            scf::YieldOp::create(rewriter, loc, next);
          });
      rewriter.replaceOp(op, forOp.getResults());
      return success();
    }
    if (auto pad = dyn_cast<tensor::PadOp>(op)) {
      auto outputType = cast<RankedTensorType>(pad.getResult().getType());
      auto inputType = cast<RankedTensorType>(pad.getSource().getType());
      if (!inputType.hasStaticShape() || !outputType.hasStaticShape() ||
          operands.size() != 1)
        return failure();
      Block &body = pad.getRegion().front();
      for (BlockArgument arg : body.getArguments())
        if (!arg.use_empty())
          return failure();
      SmallVector<Value> dummy(body.getNumArguments(), index(0));
      Value padding = scalarRegion(body, dummy, {})[0];
      Value output = vector::BroadcastOp::create(rewriter, loc,
                                                 flatType(outputType), padding);
      auto low = pad.getStaticLow();
      auto forOp = scf::ForOp::create(
          rewriter, loc, index(0), index(inputType.getNumElements()), index(1),
          ValueRange{output},
          [&](OpBuilder &, Location, Value iv, ValueRange iterArgs) {
            Value remaining = iv;
            SmallVector<Value> coords(inputType.getRank());
            for (int i = inputType.getRank() - 1; i >= 0; --i) {
              Value coord = arith::RemUIOp::create(
                  rewriter, loc, remaining, index(inputType.getDimSize(i)));
              remaining = arith::DivUIOp::create(
                  rewriter, loc, remaining, index(inputType.getDimSize(i)));
              coords[i] =
                  arith::AddIOp::create(rewriter, loc, coord, index(low[i]));
            }
            Value next = insert(extract(operands[0], iv), iterArgs[0],
                                linearIndex(coords, outputType.getShape()));
            scf::YieldOp::create(rewriter, loc, next);
          });
      rewriter.replaceOp(op, forOp.getResults());
      return success();
    }
    return failure();
  }
};

struct LowerSmallTensorValuesPass
    : mlir::enzyme::impl::LowerSmallTensorValuesPassBase<
          LowerSmallTensorValuesPass> {
  using Base::Base;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, scf::SCFDialect,
                    vector::VectorDialect, ub::UBDialect>();
  }
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    if (maxElements < 1) {
      getOperation().emitError("max-elements must be positive");
      signalPassFailure();
      return;
    }
    auto supported = [&](Type type) {
      auto tensor = dyn_cast<RankedTensorType>(type);
      auto vector = dyn_cast<VectorType>(type);
      if (isa<IndexType>(type))
        return true;
      Type element = tensor   ? tensor.getElementType()
                     : vector ? vector.getElementType()
                              : type;
      bool scalar = element.isF32() || element.isF64();
      if (auto integer = dyn_cast<IntegerType>(element))
        scalar = llvm::is_contained(ArrayRef<unsigned>{1, 8, 16, 32, 64},
                                    integer.getWidth());
      if (!scalar)
        return false;
      if (!tensor && !vector)
        return true;
      if ((tensor && (tensor.getEncoding() || !tensor.hasStaticShape())) ||
          (vector && vector.isScalable()))
        return false;
      int64_t count = 1;
      for (int64_t size : tensor ? tensor.getShape() : vector.getShape()) {
        if (size <= 0 || size > maxElements / count)
          return false;
        count *= size;
      }
      return true;
    };
    WalkResult checked = getOperation().walk([&](Operation *op) {
      bool valid = llvm::all_of(op->getOperandTypes(), supported) &&
                   llvm::all_of(op->getResultTypes(), supported);
      for (Region &region : op->getRegions())
        for (Block &block : region)
          valid = valid && llvm::all_of(block.getArgumentTypes(), supported);
      if (valid)
        return WalkResult::advance();
      op->emitError("small GPU profile requires unencoded static nonempty "
                    "tensors with supported element types and at most ")
          << maxElements.getValue() << " elements";
      return WalkResult::interrupt();
    });
    if (checked.wasInterrupted()) {
      signalPassFailure();
      return;
    }
    if (auto entry =
            getOperation().lookupSymbol<func::FuncOp>(entryFunction.getValue()))
      entry->setAttr("enzymexla.small_program_type",
                     TypeAttr::get(entry.getFunctionType()));
    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    converter.addConversion([](IntegerType type) -> Type {
      return IntegerType::get(type.getContext(), type.getWidth());
    });
    converter.addConversion([](RankedTensorType type) -> std::optional<Type> {
      if (!type.hasStaticShape() || type.getNumElements() == 0)
        return Type();
      return flatType(type);
    });
    auto materialize = [](OpBuilder &builder, Type type, ValueRange values,
                          Location loc) -> Value {
      return UnrealizedConversionCastOp::create(builder, loc, type, values)
          .getResult(0);
    };
    converter.addSourceMaterialization(materialize);
    converter.addTargetMaterialization(materialize);
    RewritePatternSet patterns(ctx);
    patterns.add<Values>(converter, ctx);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    populateReturnOpTypeConversionPattern(patterns, converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    ConversionTarget target(*ctx);
    target.addLegalOp<ModuleOp>();
    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return converter.isLegal(op); });
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType()) &&
             converter.isLegal(&op.getBody());
    });
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
