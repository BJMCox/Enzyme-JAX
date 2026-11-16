//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "src/enzyme_ad/jax/Passes/Passes.h"
namespace mlir::enzyme {
#define GEN_PASS_DEF_LOWERSMALLPROGRAMCONTROLPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace mlir::enzyme
using namespace mlir;
namespace {

static SmallVector<Value> cloneValues(PatternRewriter &rewriter, Block &block,
                                      ValueRange args) {
  IRMapping mapping;
  for (auto [arg, value] : llvm::zip_equal(block.getArguments(), args))
    mapping.map(arg, value);
  for (Operation &op : block.without_terminator())
    rewriter.clone(op, mapping);
  SmallVector<Value> result;
  for (Value value : block.getTerminator()->getOperands())
    result.push_back(mapping.lookupOrDefault(value));
  return result;
}

struct While : OpRewritePattern<stablehlo::WhileOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::WhileOp op,
                                PatternRewriter &rewriter) const override {
    auto loop = scf::WhileOp::create(rewriter, op.getLoc(), op.getResultTypes(),
                                     op.getOperands());
    rewriter.inlineRegionBefore(op.getCond(), loop.getBefore(),
                                loop.getBefore().end());
    rewriter.inlineRegionBefore(op.getBody(), loop.getAfter(),
                                loop.getAfter().end());
    Block &before = loop.getBefore().front();
    Operation *condition = before.getTerminator();
    rewriter.setInsertionPoint(condition);
    Value predicate = tensor::ExtractOp::create(rewriter, op.getLoc(),
                                                condition->getOperand(0));
    rewriter.replaceOpWithNewOp<scf::ConditionOp>(condition, predicate,
                                                  before.getArguments());
    Operation *bodyReturn = loop.getAfter().front().getTerminator();
    rewriter.setInsertionPoint(bodyReturn);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(bodyReturn,
                                              bodyReturn->getOperands());
    rewriter.replaceOp(op, loop.getResults());
    return success();
  }
};

struct If : OpRewritePattern<stablehlo::IfOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::IfOp op,
                                PatternRewriter &rewriter) const override {
    Value predicate =
        tensor::ExtractOp::create(rewriter, op.getLoc(), op.getPred());
    auto result = scf::IfOp::create(
        rewriter, op.getLoc(), predicate,
        [&](OpBuilder &, Location) {
          scf::YieldOp::create(
              rewriter, op.getLoc(),
              cloneValues(rewriter, op.getTrueBranch().front(), {}));
        },
        [&](OpBuilder &, Location) {
          scf::YieldOp::create(
              rewriter, op.getLoc(),
              cloneValues(rewriter, op.getFalseBranch().front(), {}));
        });
    rewriter.replaceOp(op, result.getResults());
    return success();
  }
};

struct Case : OpRewritePattern<stablehlo::CaseOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::CaseOp op,
                                PatternRewriter &rewriter) const override {
    Value selector =
        tensor::ExtractOp::create(rewriter, op.getLoc(), op.getIndex());
    std::function<SmallVector<Value>(unsigned)> branch;
    branch = [&](unsigned i) -> SmallVector<Value> {
      if (i + 1 == op.getBranches().size())
        return cloneValues(rewriter, op.getBranches()[i].front(), {});
      Value constant = arith::ConstantIntOp::create(rewriter, op.getLoc(),
                                                    selector.getType(), i);
      Value matches = arith::CmpIOp::create(
          rewriter, op.getLoc(), arith::CmpIPredicate::eq, selector, constant);
      auto result = scf::IfOp::create(
          rewriter, op.getLoc(), matches,
          [&](OpBuilder &, Location) {
            scf::YieldOp::create(
                rewriter, op.getLoc(),
                cloneValues(rewriter, op.getBranches()[i].front(), {}));
          },
          [&](OpBuilder &, Location) {
            scf::YieldOp::create(rewriter, op.getLoc(), branch(i + 1));
          });
      return SmallVector<Value>(result.getResults());
    };
    rewriter.replaceOp(op, branch(0));
    return success();
  }
};

// StableHLO's ReduceOp_RemoveNoop identity also covers scalar checkify checks.
struct EmptyReduction : OpRewritePattern<stablehlo::ReduceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getDimensions().empty())
      return failure();
    for (auto [input, output] :
         llvm::zip_equal(op.getInputs(), op.getResults()))
      if (input.getType() != output.getType())
        return failure();
    rewriter.replaceOp(op, op.getInputs());
    return success();
  }
};

// A single indexed complete window, including a scalar update. Preserve dropped
// out-of-bounds updates instead of turning them into a clamped update slice.
struct Scatter : OpRewritePattern<stablehlo::ScatterOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::ScatterOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1)
      return failure();
    Value input = op.getInputs()[0], updates = op.getUpdates()[0],
          indices = op.getScatterIndices();
    auto inputType = cast<RankedTensorType>(input.getType());
    auto updatesType = cast<RankedTensorType>(updates.getType());
    auto indexType = cast<RankedTensorType>(indices.getType());
    auto dims = op.getScatterDimensionNumbers();
    if (!inputType.hasStaticShape() || !updatesType.hasStaticShape() ||
        !indexType.hasStaticShape() || indexType.getNumElements() != 1 ||
        dims.getScatterDimsToOperandDims().size() != 1 ||
        !dims.getInputBatchingDims().empty() ||
        !dims.getScatterIndicesBatchingDims().empty())
      return failure();
    int64_t axis = dims.getScatterDimsToOperandDims()[0];
    if (dims.getInsertedWindowDims() != ArrayRef<int64_t>{axis} ||
        dims.getUpdateWindowDims().size() != inputType.getRank() - 1)
      return failure();
    SmallVector<int64_t> window;
    for (int64_t d = 0; d < inputType.getRank(); ++d)
      if (d != axis)
        window.push_back(inputType.getDimSize(d));
    SmallVector<bool> used(updatesType.getRank(), false);
    for (auto [i, d] : llvm::enumerate(dims.getUpdateWindowDims())) {
      if (updatesType.getDimSize(d) != window[i])
        return failure();
      used[d] = true;
    }
    for (auto [d, size] : llvm::enumerate(updatesType.getShape()))
      if (!used[d] && size != 1)
        return failure();
    auto loc = op.getLoc();
    auto constant = [&](int64_t value) -> Value {
      return arith::ConstantIndexOp::create(rewriter, loc, value);
    };
    SmallVector<Value> zeroIndices(indexType.getRank(), constant(0));
    Value row = tensor::ExtractOp::create(rewriter, loc, indices, zeroIndices);
    row =
        arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), row);
    Value valid =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult, row,
                              constant(inputType.getDimSize(axis)));
    auto result = scf::IfOp::create(
        rewriter, loc, valid,
        [&](OpBuilder &, Location) {
          Value lower = constant(0);
          Value upper = constant(updatesType.getNumElements());
          Value step = constant(1);
          auto loop = scf::ForOp::create(
              rewriter, loc, lower, upper, step, ValueRange{input},
              [&](OpBuilder &, Location, Value iv, ValueRange carried) {
                Value remaining = iv;
                SmallVector<Value> updateCoords(updatesType.getRank());
                for (int d = updatesType.getRank() - 1; d >= 0; --d) {
                  updateCoords[d] = arith::RemUIOp::create(
                      rewriter, loc, remaining,
                      constant(updatesType.getDimSize(d)));
                  remaining = arith::DivUIOp::create(
                      rewriter, loc, remaining,
                      constant(updatesType.getDimSize(d)));
                }
                SmallVector<Value> inputCoords;
                unsigned w = 0;
                for (int d = 0; d < inputType.getRank(); ++d)
                  inputCoords.push_back(
                      d == axis
                          ? row
                          : updateCoords[dims.getUpdateWindowDims()[w++]]);
                Value oldScalar = tensor::ExtractOp::create(
                    rewriter, loc, carried[0], inputCoords);
                Value newScalar = tensor::ExtractOp::create(
                    rewriter, loc, updates, updateCoords);
                auto scalarType =
                    RankedTensorType::get({}, inputType.getElementType());
                Value oldTensor = tensor::FromElementsOp::create(
                    rewriter, loc, scalarType, oldScalar);
                Value newTensor = tensor::FromElementsOp::create(
                    rewriter, loc, scalarType, newScalar);
                Value yielded =
                    cloneValues(rewriter, op.getUpdateComputation().front(),
                                ValueRange{oldTensor, newTensor})[0];
                yielded = tensor::ExtractOp::create(rewriter, loc, yielded);
                Value next = tensor::InsertOp::create(rewriter, loc, yielded,
                                                      carried[0], inputCoords);
                scf::YieldOp::create(rewriter, loc, next);
              });
          scf::YieldOp::create(rewriter, loc, loop.getResults());
        },
        [&](OpBuilder &, Location) {
          scf::YieldOp::create(rewriter, loc, input);
        });
    rewriter.replaceOp(op, result.getResults());
    return success();
  }
};

struct LowerSmallProgramControlPass
    : mlir::enzyme::impl::LowerSmallProgramControlPassBase<
          LowerSmallProgramControlPass> {
  using Base::Base;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, scf::SCFDialect, tensor::TensorDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    WalkResult checked = module.walk([&](sdy::ShardingConstraintOp op) {
      auto mesh = op.getSharding().getMesh(op);
      if (mesh && mesh.empty() && op.getSharding().emptyAxes())
        return WalkResult::advance();
      op.emitError(
          "small GPU profile supports only empty-mesh sharding constraints");
      return WalkResult::interrupt();
    });
    if (checked.wasInterrupted())
      return signalPassFailure();
    // Typed JAX keys carry this placeholder annotation even on one device.
    module.walk([&](sdy::ShardingConstraintOp op) {
      op.replaceAllUsesWith(op.getInput());
      op.erase();
    });
    module.walk([&](sdy::MeshOp op) {
      if (op.getMesh().empty() && SymbolTable::symbolKnownUseEmpty(op, module))
        op.erase();
    });
    RewritePatternSet patterns(&getContext());
    patterns.add<While, If, Case, EmptyReduction, Scatter>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
