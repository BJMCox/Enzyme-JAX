#pragma once

#include "Enzyme/MLIR/Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Enzyme/MLIR/Interfaces/AutoDiffOpInterface.h"
#include "Enzyme/MLIR/Interfaces/GradientUtils.h"
#include "Enzyme/MLIR/Interfaces/GradientUtilsReverse.h"
#include "Enzyme/MLIR/Passes/EnzymeBatchPass.h"
#include "Enzyme/MLIR/Passes/RemovalUtils.h"

#include "src/enzyme_ad/jax/Implementations/WhileLoopInfo.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::stablehlo;

namespace details {

static Value makeIntegerConstant(Location loc, OpBuilder &builder, Type type,
                                 int64_t val) {
  auto unrankedTensorType = RankedTensorType::get({}, type);
  return ConstantOp::create(
             builder, loc, unrankedTensorType,
             SplatElementsAttr::get(
                 unrankedTensorType,
                 ArrayRef<Attribute>(IntegerAttr::get(type, val))))
      .getResult();
}

static Value makeI64Constant(Location loc, OpBuilder &builder, int64_t val) {
  return makeIntegerConstant(loc, builder, builder.getI64Type(), val);
}

} // namespace details

inline mlir::TensorType applyBatchSizes(mlir::Type Ty,
                                        llvm::ArrayRef<int64_t> batchSizes) {
  auto T = cast<TensorType>(Ty);
  SmallVector<int64_t> shape(batchSizes.begin(), batchSizes.end());
  shape.append(T.getShape().begin(), T.getShape().end());
  auto T2 = T.clone(shape);
  return T2;
}

inline void getAllReferences(SmallVector<Value> &refs, Operation *op,
                             Region *ref) {
  for (auto operand : op->getOperands()) {
    if (operand.getParentRegion()->isAncestor(ref))
      refs.push_back(operand);
  }

  for (auto &reg : op->getRegions()) {
    for (auto &childOp : reg.getOps()) {
      getAllReferences(refs, &childOp, ref);
    }
  }
}

inline void batchCloneBlock(Block *srcBlock, Block *destBlock,
                            IRMapping &mapper, ArrayRef<int64_t> batchSizes) {
  for (auto arg : srcBlock->getArguments()) {
    auto batched = destBlock->addArgument(
        applyBatchSizes(arg.getType(), batchSizes), arg.getLoc());
    mapper.map(arg, batched);
  }

  OpBuilder builder(destBlock, destBlock->end());

  std::map<enzyme::batchutils::BatchCacheKey, FunctionOpInterface>
      batchedFunctionCache;
  enzyme::batchutils::batchCloneBlock(builder, srcBlock, mapper, batchSizes,
                                      batchedFunctionCache, false);
}

inline LogicalResult tryToBatchInner(Operation *src, OpBuilder &builder,
                                     IRMapping &mapper,
                                     ArrayRef<int64_t> batchSizes) {
  if (auto loop = dyn_cast<stablehlo::WhileOp>(src)) {
    // Fixed scans advance all lanes together. Keep the induction variable
    // scalar so body slices stay slices, rather than per-lane gathers.
    enzyme::WhileLoopInfo info(loop);
    if (failed(info.computeInfo()) || !info.isConstant() ||
        llvm::is_contained(batchSizes, 0))
      return failure();
    bool safe = true;
    loop.walk([&](Operation *op) {
      auto tensor = [](Type type) {
        auto ty = dyn_cast<RankedTensorType>(type);
        return ty && ty.hasStaticShape() && !ty.getEncoding() &&
               (ty.getElementType().isIntOrFloat() ||
                isa<ComplexType>(ty.getElementType()));
      };
      if (op->getName().getDialectNamespace() != "stablehlo" ||
          !llvm::all_of(op->getOperandTypes(), tensor) ||
          !llvm::all_of(op->getResultTypes(), tensor) ||
          !(op->hasTrait<OpTrait::Elementwise>() ||
            isa<WhileOp, ReturnOp, ConstantOp, ReshapeOp, BroadcastInDimOp,
                TransposeOp, SliceOp, DynamicSliceOp, DynamicUpdateSliceOp,
                ConcatenateOp, DotGeneralOp, GatherOp, IotaOp,
                GetDimensionSizeOp, ReverseOp>(op)) ||
          !isMemoryEffectFree(op))
        safe = false;
    });
    if (!safe)
      return failure();
    auto index = info.getArgNumber();
    Value iv = loop.getBody().front().getArgument(index);
    Value update = loop.getBody().front().getTerminator()->getOperand(index);
    // The range analysis also follows converts. Narrowing an induction value
    // changes its wraparound, so only replace a direct same-type recurrence.
    auto *step = update.getDefiningOp();
    if (!step || update.getType() != iv.getType() ||
        !((isa<AddOp>(step) &&
           ((step->getOperand(0) == iv &&
             matchPattern(step->getOperand(1), m_Constant())) ||
            (step->getOperand(1) == iv &&
             matchPattern(step->getOperand(0), m_Constant())))) ||
          (isa<SubtractOp>(step) && step->getOperand(0) == iv &&
           matchPattern(step->getOperand(1), m_Constant()))))
      return failure();
    auto loc = src->getLoc();
    auto indexType = cast<RankedTensorType>(loop->getOperand(index).getType());
    auto constant = [&](OpBuilder &b, int64_t value) -> Value {
      return details::makeIntegerConstant(loc, b, indexType.getElementType(),
                                          value);
    };
    SmallVector<Value> operands;
    for (auto [i, operand] : llvm::enumerate(loop.getOperands()))
      operands.push_back(i == index
                             ? constant(builder, *info.getConstantStart())
                             : mapper.lookup(operand));
    auto batched = WhileOp::create(builder, loc, operands);
    auto *cond = new Block();
    auto *body = new Block();
    batched.getCond().push_back(cond);
    batched.getBody().push_back(body);
    for (auto operand : operands) {
      cond->addArgument(operand.getType(), loc);
      body->addArgument(operand.getType(), loc);
    }
    {
      OpBuilder b(cond, cond->end());
      auto more = CompareOp::create(b, loc, cond->getArgument(index),
                                    constant(b, *info.getConstantLimit()),
                                    ComparisonDirection::LT);
      auto original = loop.getCond()
                          .front()
                          .getTerminator()
                          ->getOperand(0)
                          .getDefiningOp<CompareOp>();
      more->setAttrs(original->getAttrs());
      ReturnOp::create(b, loc, ValueRange(more));
    }
    {
      OpBuilder b(body, body->end());
      IRMapping bodyMapper = mapper;
      for (auto [i, arg] :
           llvm::enumerate(loop.getBody().front().getArguments())) {
        Value value = body->getArgument(i);
        if (i == index)
          value = BroadcastInDimOp::create(
              b, loc, applyBatchSizes(arg.getType(), batchSizes), value,
              b.getDenseI64ArrayAttr({}));
        bodyMapper.map(arg, value);
      }
      std::map<enzyme::batchutils::BatchCacheKey, FunctionOpInterface> cache;
      enzyme::batchutils::batchCloneBlock(b, &loop.getBody().front(),
                                          bodyMapper, batchSizes, cache, false);
      auto *term = body->getTerminator();
      b.setInsertionPoint(term);
      auto next = AddOp::create(b, loc, body->getArgument(index),
                                constant(b, *info.getConstantStep()));
      term->setOperand(index, next);
    }
    for (auto [i, result] : llvm::enumerate(loop.getResults())) {
      Value value = batched.getResult(i);
      if (i == index)
        value = BroadcastInDimOp::create(
            builder, loc, applyBatchSizes(result.getType(), batchSizes), value,
            builder.getDenseI64ArrayAttr({}));
      mapper.map(result, value);
    }
    return success();
  }

  if (auto ifOp = dyn_cast<stablehlo::IfOp>(src)) {
    auto predBroadcast =
        mapper.lookup(ifOp.getPred()).getDefiningOp<BroadcastInDimOp>();
    if (predBroadcast && predBroadcast.isSimpleBroadcast() &&
        predBroadcast.getBroadcastDimensions().size() == batchSizes.size()) {
      // %pred = broadcast_in_dim %0
      // if %0 {} {}
      SmallVector<Type> results;
      results.reserve(src->getNumResults());
      for (auto resTy : src->getResultTypes()) {
        results.push_back(applyBatchSizes(resTy, batchSizes));
      }
      auto newIf = stablehlo::IfOp::create(builder, src->getLoc(), results,
                                           predBroadcast.getOperand());
      newIf.getTrueBranch().push_back(new Block());
      newIf.getFalseBranch().push_back(new Block());

      batchCloneBlock(&ifOp.getTrueBranch().front(),
                      &newIf.getTrueBranch().front(), mapper, batchSizes);
      batchCloneBlock(&ifOp.getFalseBranch().front(),
                      &newIf.getFalseBranch().front(), mapper, batchSizes);

      for (auto &&[oldRes, newRes] :
           llvm::zip(ifOp->getResults(), newIf->getResults())) {
        mapper.map(oldRes, newRes);
      }

      return success();
    }

    auto iszero = matchPattern(ifOp.getPred(), m_Zero());
    auto isone = matchPattern(ifOp.getPred(), m_One());

    if (!iszero && !isone)
      return failure();

    auto &reg = isone ? ifOp.getTrueBranch() : ifOp.getFalseBranch();

    assert(reg.hasOneBlock());  // stablehlo.if only allows 1 or 0 block in the
    auto *block = &reg.front(); // regions

    batchCloneBlock(block, builder.getInsertionBlock(), mapper, batchSizes);
    auto term = builder.getInsertionBlock()->getTerminator();

    for (auto &&[result, operand] :
         llvm::zip(src->getResults(), term->getOperands())) {
      mapper.map(result, operand);
    }

    term->erase();

    return success();
  }

  return failure();
}

inline LogicalResult genericCreateBatch(Operation *src, OpBuilder &builder,
                                        IRMapping &mapper,
                                        ArrayRef<int64_t> batchSizes) {
  if (tryToBatchInner(src, builder, mapper, batchSizes).succeeded())
    return success();

  SmallVector<Value> operands;
  operands.reserve(src->getNumOperands());

  getAllReferences(operands, src, src->getParentRegion());

  SmallVector<Value> whileOperands;
  whileOperands.reserve(src->getNumResults() + 1);
  whileOperands.push_back(details::makeI64Constant(src->getLoc(), builder, 0));

  for (auto res : src->getResults()) {
    auto Ty = cast<TensorType>(res.getType());
    SmallVector<int64_t> shape(batchSizes.begin(), batchSizes.end());
    shape.append(Ty.getShape().begin(), Ty.getShape().end());
    auto T2 = cast<AutoDiffTypeInterface>(Ty.clone(shape));
    auto defaultValue = T2.createNullValue(builder, src->getLoc());
    mapper.map(res, defaultValue);
    whileOperands.push_back(defaultValue);
  }

  auto ndims = batchSizes.size();

  SmallVector<int64_t> batchStrides;
  batchStrides.reserve(ndims);
  SmallVector<Value> startIndices;
  startIndices.reserve(ndims);

  int64_t N = 1;
  for (auto batchSize : batchSizes) {
    batchStrides.push_back(N);
    N *= batchSize;
  }

  auto whileOp = WhileOp::create(builder, src->getLoc(), whileOperands);

  auto whileCond = new Block();
  auto whileBody = new Block();

  whileOp.getCond().push_back(whileCond);
  whileOp.getBody().push_back(whileBody);

  {
    OpBuilder condBuilder(whileCond, whileCond->end());

    for (auto operand : whileOperands) {
      whileCond->addArgument(operand.getType(), src->getLoc());
    }

    ReturnOp::create(
        condBuilder, src->getLoc(),
        ValueRange(CompareOp::create(
            condBuilder, src->getLoc(), whileCond->getArgument(0),
            details::makeI64Constant(src->getLoc(), condBuilder, N),
            ComparisonDirection::LT)));
  }

  {
    OpBuilder bodyBuilder(whileBody, whileBody->end());

    for (auto operand : whileOperands) {
      whileBody->addArgument(operand.getType(), src->getLoc());
    }

    SmallVector<Value> whileBodyOutputs;
    whileBodyOutputs.reserve(whileBody->getNumArguments());

    whileBodyOutputs.push_back(
        AddOp::create(bodyBuilder, src->getLoc(), whileBody->getArgument(0),
                      details::makeI64Constant(src->getLoc(), bodyBuilder, 1)));

    for (int d = 0; d < ndims; ++d) {
      // auto idx = (i / batchStrides[d]) % batchSizes[d];
      Value stride =
          details::makeI64Constant(src->getLoc(), bodyBuilder, batchStrides[d]);
      Value quotient = DivOp::create(bodyBuilder, src->getLoc(),
                                     whileBody->getArgument(0), stride);
      Value size =
          details::makeI64Constant(src->getLoc(), bodyBuilder, batchSizes[d]);
      auto idx = RemOp::create(bodyBuilder, src->getLoc(), quotient, size);

      startIndices.push_back(idx);
    }

    auto zeroIdx = details::makeI64Constant(src->getLoc(), bodyBuilder, 0);

    IRMapping origToUnbatch;
    for (auto operand : operands) {
      auto batched = mapper.lookup(operand);

      auto Ty = cast<TensorType>(operand.getType());
      SmallVector<int64_t> shape(ndims, 1);
      shape.append(Ty.getShape().begin(), Ty.getShape().end());
      auto sliceTy = Ty.clone(shape);

      SmallVector<Value> operandStartIndices;
      operandStartIndices.append(startIndices.begin(), startIndices.end());
      for (auto i = 0; i < Ty.getShape().size(); i++)
        operandStartIndices.push_back(zeroIdx);

      auto sliceOp = stablehlo::DynamicSliceOp::create(
          bodyBuilder, src->getLoc(), sliceTy, batched, operandStartIndices,
          shape);

      auto reshapeOp = stablehlo::ReshapeOp::create(
          bodyBuilder, src->getLoc(), operand.getType(), sliceOp->getResult(0));

      origToUnbatch.map(operand, reshapeOp->getResult(0));
    }

    auto newOp = bodyBuilder.clone(*src, origToUnbatch);

    for (auto &&[idx, origRes, newRes] :
         llvm::enumerate(src->getResults(), newOp->getResults())) {
      auto batched = whileBody->getArgument(idx + 1);

      auto Ty = cast<TensorType>(newRes.getType());
      SmallVector<int64_t> shape(ndims, 1);
      shape.append(Ty.getShape().begin(), Ty.getShape().end());
      auto reshapeTy = Ty.clone(shape);

      auto reshapeOp = stablehlo::ReshapeOp::create(bodyBuilder, src->getLoc(),
                                                    reshapeTy, newRes);

      SmallVector<Value> operandStartIndices;
      operandStartIndices.append(startIndices.begin(), startIndices.end());
      for (int i = 0; i < Ty.getShape().size(); ++i)
        operandStartIndices.push_back(zeroIdx);

      auto update = stablehlo::DynamicUpdateSliceOp::create(
          bodyBuilder, src->getLoc(), batched, reshapeOp, operandStartIndices);

      whileBodyOutputs.push_back(update);
    }

    ReturnOp::create(bodyBuilder, src->getLoc(), whileBodyOutputs);
  }

  for (auto oldRes : src->getOpResults()) {
    mapper.map(oldRes, whileOp->getResult(oldRes.getResultNumber() + 1));
  }

  return success();
}

template <typename OpTy>
struct SHLOGenericBatchOpInterface
    : public BatchOpInterface::ExternalModel<SHLOGenericBatchOpInterface<OpTy>,
                                             OpTy> {
  LogicalResult createBatch(Operation *src, OpBuilder &builder,
                            IRMapping &mapper,
                            ArrayRef<int64_t> batchSizes) const {
    return genericCreateBatch(src, builder, mapper, batchSizes);
  }
};

template <typename OpTy>
struct HLOConstantOpBatchInterface
    : public BatchOpInterface::ExternalModel<HLOConstantOpBatchInterface<OpTy>,
                                             OpTy> {

  mlir::LogicalResult createBatch(Operation *src, OpBuilder &builder,
                                  IRMapping &mapper,
                                  ArrayRef<int64_t> batchSizes) const {
    auto constOp = cast<OpTy>(src);

    auto T = cast<TensorType>(constOp.getType());
    SmallVector<int64_t> shape(batchSizes.begin(), batchSizes.end());
    shape.append(T.getShape().begin(), T.getShape().end());
    auto Ty = T.clone(shape);

    // If splatted attr then we can easily batch it
    auto eattr = cast<DenseElementsAttr>(constOp.getValue());
    if (eattr.isSplat()) {
      auto splatAttr = cast<SplatElementsAttr>(constOp.getValue());
      auto newSplattedConstOp = OpTy::create(
          builder, constOp->getLoc(), Ty,
          cast<ElementsAttr>(splatAttr.resizeSplat(cast<ShapedType>(Ty))));
      mapper.map(src->getResult(0), newSplattedConstOp->getResult(0));
      return success();
    }

    // otherwise do a broadcast in dim
    SmallVector<int64_t> mapping(T.getShape().size());
    std::iota(mapping.begin(), mapping.end(), batchSizes.size());

    auto constOpCloned = builder.clone(*constOp);
    auto bcastOp = BroadcastInDimOp::create(
        builder, src->getLoc(), Ty, constOpCloned->getResult(0),
        builder.getDenseI64ArrayAttr(mapping));
    mapper.map(src->getResult(0), bcastOp->getResult(0));
    return success();
  }
};
