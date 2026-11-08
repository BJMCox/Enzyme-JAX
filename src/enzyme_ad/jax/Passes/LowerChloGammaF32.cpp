//===- LowerChloGammaF32.cpp - Accurate float32 lgamma and digamma --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Rewrites float32 chlo.lgamma and chlo.digamma into StableHLO.
//
// XLA's CHLO legalization evaluates both functions with a Lanczos series. In
// float32 its error peaks near the positive roots: up to about 230 ulp for
// lgamma and 80 ulp for digamma on (1, 10). Enzyme differentiates
// chlo.lgamma into chlo.digamma, so the gradient inherits the second error.
//
// lgamma follows the float32 algorithm of JAX 0.12 (jax/_src/lax/lgamma.py):
// factored minimax polynomials on [0.5, 4], a Stirling series above 4, Euler
// reflection below 0.5, and Cayley-transformed root windows on (-10, -2).
//
// digamma reduces x >= 0.5 to [1, 2] by the recurrence psi(x + 1) =
// psi(x) + 1/x, and evaluates psi(y) = (y - x0) P(y - 1) there, with x0 the
// positive root of psi split into two floats. Above 6 it uses the asymptotic
// series. Below 0.5 it reflects: psi(x) = psi(1 - x) - pi cot(pi x).
//
// Run this pass after differentiation: the rewrite has no derivative rule of
// its own, so AD must see the chlo ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <limits>
#include <utility>

#define DEBUG_TYPE "lower-chlo-gamma-f32"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERCHLOGAMMAF32PASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace {

// Float32 constants copied bit for bit from JAX 0.12 lgamma.py.
constexpr float kLogQHi[] = {0x1.555556p-2f,  -0x1.fffff2p-3f, 0x1.999990p-3f,
                             -0x1.555c5cp-3f, 0x1.249b48p-3f,  -0x1.fe01acp-4f,
                             0x1.c37530p-4f,  -0x1.b33c1ap-4f, 0x1.b2f136p-4f,
                             -0x1.044848p-4f};
constexpr float kPosHi0515[] = {
    0x1.2788d0p-1f, -0x1.f64652p-3f, 0x1.3e53d6p-3f, -0x1.d7aaaap-4f,
    0x1.79f672p-4f, -0x1.3bdb70p-4f, 0x1.0eabb8p-4f, -0x1.f7b950p-5f,
    0x1.dddba8p-5f, -0x1.f647f0p-6f, 0x1.127e1ep-6f, -0x1.62e36cp-4f,
    0x1.837a10p-4f};
constexpr float kPosHi154[] = {
    0x1.72c20ap-2f,  -0x1.0d3a36p-4f,  0x1.06a644p-6f,  -0x1.21a452p-8f,
    0x1.55c08cp-10f, -0x1.a385ccp-12f, 0x1.05e352p-13f, -0x1.5f79c0p-15f,
    0x1.04bf3ap-16f, -0x1.ce70fep-19f, 0x1.ccff74p-23f, -0x1.feb34ep-21f,
    0x1.117724p-21f};
constexpr float kStirlingHi[] = {0x1.555556p-4f, -0x1.6c16c2p-9f,
                                 0x1.a01a02p-11f, -0x1.381382p-11f,
                                 0x1.b951e2p-11f};
constexpr float kSincHi[] = {-0x1.a51a66p+0f, -0x1.151306p-1f, -0x1.5b4f0ap-2f,
                             -0x1.ff7a74p-3f, -0x1.b6a08ep-3f, -0x1.72eacep-4f,
                             -0x1.549150p-2f};
constexpr float kNegTable[14][16] = {
    {-0x1.d3fe4cp-2f, 0x1.000000p+0f, -0x1.260dbcp-3f, 0x1.025f7ap-2f,
     -0x1.4273c2p-5f, 0x1.6e3aaep-5f, -0x1.0d4afep-7f, 0x1.1510b2p-7f,
     -0x1.6b258ap-10f, 0x1.6d0a6ep-10f, -0x1.9fef70p-13f, 0x1.a044a4p-13f,
     -0x1.a0145ap-16f, 0x1.a01faap-16f, -0x1.71dda4p-19f, 0x1.71ded0p-19f},
    {0x1.ff0794p-27f, 0x0.0p+0f, -0x1.3cb35ep-28f, 0x1.e42700p-27f,
     -0x1.995808p-30f, 0x1.e80d48p-34f, -0x1.6db2f4p-35f, 0x1.15037cp-34f,
     0x1.06e6f8p-35f, 0x1.7e50c0p-39f, 0x1.e17f4ep-42f, -0x1.bbe00ap-40f,
     0x1.a1fb90p-47f, -0x1.cf752ap-42f, 0x1.214d2cp-47f, 0x1.7c026ap-44f},
    {0x1.051eb8p-2f, -0x1.000000p+0f, 0x1.5c28f6p-2f, 0x1.7ae148p-2f,
     0x1.ae147ap-2f, 0x1.ae147ap-2f, 0x1.ae147ap-2f, 0x1.ae147ap-2f,
     0x1.ae147ap-2f, 0x1.ae147ap-2f, 0x1.ae147ap-2f, 0x1.ae147ap-2f,
     0x1.ae147ap-2f, 0x1.ae147ap-2f, 0x1.ae147ap-2f, 0x1.ae147ap-2f},
    {-0x1.62a56ap+0f, 0x0.0p+0f, -0x1.1e0956p+1f, -0x1.ee9dc8p-1f,
     -0x1.0df532p+1f, -0x1.da61f8p+0f, -0x1.038922p+1f, -0x1.f8800cp+0f,
     -0x1.00a9a2p+1f, -0x1.fea962p+0f, -0x1.001a30p+1f, -0x1.ffcb8cp+0f,
     -0x1.00037ap+1f, -0x1.fff90ap+0f, -0x1.000068p+1f, -0x1.ffff30p+0f},
    {0x1.563c7ep+1f, 0x0.0p+0f, -0x1.ac4506p-4f, 0x1.78eadap+0f,
     -0x1.978ccep-4f, 0x1.46261ap-3f, -0x1.bdaf66p-6f, 0x1.e745cap-6f,
     -0x1.527876p-8f, 0x1.576a92p-8f, -0x1.a2edb8p-11f, 0x1.a3c900p-11f,
     -0x1.bd5088p-14f, 0x1.bd6d28p-14f, -0x1.a1ea72p-17f, 0x1.a19bf6p-17f},
    {0x1.69eae6p+2f, 0x0.0p+0f, -0x1.4414a8p-1f, 0x1.540b04p+0f,
     -0x1.835dfap-1f, -0x1.faeb78p-2f, -0x1.630c02p-1f, -0x1.45e118p-1f,
     -0x1.57f8e6p-1f, -0x1.52a51ap-1f, -0x1.55be3ap-1f, -0x1.54ec8ap-1f,
     -0x1.556370p-1f, -0x1.55479ap-1f, -0x1.555720p-1f, -0x1.5553e0p-1f},
    {0x1.b4c28ap+3f, 0x0.0p+0f, 0x1.798d2ap-3f, 0x1.54e862p+1f, -0x1.48b8bep-4f,
     0x1.78cfb0p-3f, -0x1.afd00ap-6f, 0x1.f5a8d0p-6f, -0x1.50cf98p-8f,
     0x1.58ec28p-8f, -0x1.a29210p-11f, 0x1.a3f366p-11f, -0x1.bce868p-14f,
     0x1.bd1dacp-14f, -0x1.6abd60p-17f, 0x1.778ed6p-17f},
    {0x1.d0e58ep+4f, 0x0.0p+0f, -0x1.85bf90p-5f, 0x1.958884p+1f,
     -0x1.e1b2f6p-2f, -0x1.a0c928p-3f, -0x1.b407cap-2f, -0x1.79aed6p-2f,
     -0x1.9ebe9cp-2f, -0x1.9418bcp-2f, -0x1.9a514ap-2f, -0x1.98ae36p-2f,
     -0x1.999c24p-2f, -0x1.996496p-2f, -0x1.99856ap-2f, -0x1.997e94p-2f},
    {0x1.02c550p+6f, 0x0.0p+0f, 0x1.1b6524p-1f, 0x1.35078ap+2f, -0x1.efbe88p-5f,
     0x1.b3ca4cp-3f, -0x1.a58bb2p-6f, 0x1.0517dep-5f, -0x1.520eeep-8f,
     0x1.5d8d46p-8f, -0x1.a5c098p-11f, 0x1.a7a1f0p-11f, -0x1.c4bdf4p-14f,
     0x1.c478e4p-14f, -0x1.28da16p-15f, 0x1.ef9348p-16f},
    {0x1.208c80p+7f, 0x0.0p+0f, 0x1.0187e4p-1f, 0x1.8e6952p+2f, -0x1.5aac82p-2f,
     -0x1.e31a24p-5f, -0x1.415d7cp-2f, -0x1.060352p-2f, -0x1.2c57e4p-2f,
     -0x1.21856cp-2f, -0x1.27c90cp-2f, -0x1.261ef8p-2f, -0x1.270d6ap-2f,
     -0x1.26d3cep-2f, -0x1.26dcdap-2f, -0x1.26da88p-2f},
    {0x1.622bd2p+8f, 0x0.0p+0f, 0x1.bf8ca0p-1f, 0x1.26a72cp+3f, -0x1.58f7f8p-5f,
     0x1.8ee384p-3f, -0x1.67bbc4p-6f, 0x1.ca188ap-6f, -0x1.28bf1cp-8f,
     0x1.348dc2p-8f, -0x1.74735ap-11f, 0x1.76cd1ap-11f, -0x1.7143e2p-14f,
     0x1.75d81ep-14f, 0x1.157496p-13f, -0x1.88597ap-14f},
    {0x1.e642c0p+9f, 0x0.0p+0f, 0x1.a87578p-1f, 0x1.f41438p+3f, -0x1.d7c638p-3f,
     -0x1.8587e2p-7f, -0x1.ba279ap-3f, -0x1.58bcd8p-3f, -0x1.9c7d98p-3f,
     -0x1.8ab67ap-3f, -0x1.9653b4p-3f, -0x1.939a12p-3f, -0x1.955b08p-3f,
     -0x1.950a18p-3f, -0x1.9631f8p-3f, -0x1.95fb36p-3f},
    {0x1.07fd3cp+11f, 0x0.0p+0f, 0x1.7bd734p+1f, 0x1.8f5582p+4f,
     -0x1.a01c44p-7f, 0x1.e22b18p-2f, -0x1.3bcda0p-5f, 0x1.ceba84p-5f,
     -0x1.0d023ap-7f, 0x1.1ce97ap-7f, -0x1.518440p-10f, 0x1.53b0a0p-10f,
     -0x1.89f6f8p-13f, 0x1.83fe26p-13f, -0x1.6aaa20p-12f, 0x1.097abap-12f},
    {0x1.0ab50cp+11f, 0x0.0p+0f, 0x1.f037a8p+1f, 0x1.2b7030p+4f,
     -0x1.28c8eap-2f, 0x1.fbc9b4p-3f, -0x1.5c5ccap-2f, -0x1.ea300ep-3f,
     -0x1.37a784p-2f, -0x1.24eb4ep-2f, -0x1.2e89e8p-2f, -0x1.2ba62cp-2f,
     -0x1.2d05ecp-2f, -0x1.2c97e2p-2f, -0x1.2be376p-2f, -0x1.2bfd7ep-2f},
};
constexpr float kMaxFiniteArg = 0x1.895f1ap+121f;
constexpr float kAllIntegersBelow = -0x1.000000p+23f;
constexpr float kLn2Hi = 0x1.62e000p-1f;
constexpr float kLn2Lo = 0x1.0bfbe8p-15f;
constexpr float kHalfLn2piMinusHalf = 0x1.acfe38p-2f;
constexpr int32_t kSqrt2Over2Bits = 0x3F3504F3;

// Least-squares fit of psi(1 + t) / (1 + t - x0) at 400 Chebyshev nodes on
// t in [0, 1], degree 11, ascending powers of t.
constexpr float kDigammaCore[] = {
    0x1.4018eap+0f, -0x1.b59a56p-1f, 0x1.81420ap-1f, -0x1.6dbb16p-1f,
    0x1.64625ep-1f, -0x1.5a741cp-1f, 0x1.422ab4p-1f, -0x1.0b3becp-1f,
    0x1.68866ep-2f, -0x1.62ab76p-3f, 0x1.b71b58p-5f, -0x1.f8da9cp-8f};
// B_2k / (2k) for k = 1..5: psi(z) ~ ln z - 1/(2z) - sum_k B_2k / (2k z^2k).
constexpr float kDigammaTail[] = {0x1.555556p-4f, -0x1.111112p-7f,
                                  0x1.041042p-8f, -0x1.111112p-8f,
                                  0x1.f07c20p-8f};
// x0 = 1.46163214496836234126... = kDigammaRootHi + kDigammaRootLo.
constexpr float kDigammaRootHi = 0x1.762d86p+0f;
constexpr float kDigammaRootLo = 0x1.ab5f20p-27f;
constexpr float kPi = 0x1.921fb6p+1f;

class Builder {
public:
  Builder(OpBuilder &b, Location loc, RankedTensorType fTy)
      : b(b), loc(loc), fTy(fTy),
        iTy(RankedTensorType::get(fTy.getShape(), b.getI32Type())) {}

  Value f(float v) {
    return stablehlo::ConstantOp::create(
        b, loc, fTy, DenseElementsAttr::get(fTy, ArrayRef<float>(v)));
  }
  Value i(int32_t v) {
    return stablehlo::ConstantOp::create(
        b, loc, iTy, DenseElementsAttr::get(iTy, ArrayRef<int32_t>(v)));
  }

  Value add(Value x, Value y) { return stablehlo::AddOp::create(b, loc, x, y); }
  Value sub(Value x, Value y) {
    return stablehlo::SubtractOp::create(b, loc, x, y);
  }
  Value mul(Value x, Value y) { return stablehlo::MulOp::create(b, loc, x, y); }
  Value div(Value x, Value y) { return stablehlo::DivOp::create(b, loc, x, y); }
  Value neg(Value x) { return stablehlo::NegOp::create(b, loc, x); }
  Value abs(Value x) { return stablehlo::AbsOp::create(b, loc, x); }
  Value min(Value x, Value y) { return stablehlo::MinOp::create(b, loc, x, y); }
  Value log(Value x) { return stablehlo::LogOp::create(b, loc, x); }
  Value sin(Value x) { return stablehlo::SineOp::create(b, loc, x); }
  Value cos(Value x) { return stablehlo::CosineOp::create(b, loc, x); }
  Value floor(Value x) { return stablehlo::FloorOp::create(b, loc, x); }
  Value roundEven(Value x) {
    return stablehlo::RoundNearestEvenOp::create(b, loc, x);
  }
  Value select(Value c, Value x, Value y) {
    return stablehlo::SelectOp::create(b, loc, c, x, y);
  }
  Value cmp(Value x, Value y, stablehlo::ComparisonDirection d) {
    return stablehlo::CompareOp::create(b, loc, x, y, d);
  }
  Value lt(Value x, Value y) {
    return cmp(x, y, stablehlo::ComparisonDirection::LT);
  }
  Value le(Value x, Value y) {
    return cmp(x, y, stablehlo::ComparisonDirection::LE);
  }
  Value gt(Value x, Value y) {
    return cmp(x, y, stablehlo::ComparisonDirection::GT);
  }
  Value ge(Value x, Value y) {
    return cmp(x, y, stablehlo::ComparisonDirection::GE);
  }
  Value eq(Value x, Value y) {
    return cmp(x, y, stablehlo::ComparisonDirection::EQ);
  }
  Value ne(Value x, Value y) {
    return cmp(x, y, stablehlo::ComparisonDirection::NE);
  }
  Value land(Value x, Value y) {
    return stablehlo::AndOp::create(b, loc, x, y);
  }
  Value lor(Value x, Value y) { return stablehlo::OrOp::create(b, loc, x, y); }
  Value lnot(Value x) { return stablehlo::NotOp::create(b, loc, x); }
  Value shl(Value x, Value y) {
    return stablehlo::ShiftLeftOp::create(b, loc, x, y);
  }
  Value sra(Value x, Value y) {
    return stablehlo::ShiftRightArithmeticOp::create(b, loc, x, y);
  }
  Value clz(Value x) { return stablehlo::ClzOp::create(b, loc, x); }
  Value toBits(Value x) {
    return stablehlo::BitcastConvertOp::create(b, loc, iTy, x);
  }
  Value fromBits(Value x) {
    return stablehlo::BitcastConvertOp::create(b, loc, fTy, x);
  }
  Value toF(Value x) { return stablehlo::ConvertOp::create(b, loc, fTy, x); }
  Value toI(Value x) { return stablehlo::ConvertOp::create(b, loc, iTy, x); }

  Value horner(Value x, ArrayRef<float> coeffs) {
    Value p = f(coeffs.back());
    for (float c : llvm::reverse(coeffs.drop_back()))
      p = add(mul(p, x), f(c));
    return p;
  }
  Value horner(Value x, ArrayRef<Value> coeffs) {
    Value p = coeffs.back();
    for (Value c : llvm::reverse(coeffs.drop_back()))
      p = add(mul(p, x), c);
    return p;
  }

  // Elementwise table lookup: row[idx] for each element of idx.
  Value lookup(ArrayRef<float> row, Value idx) {
    auto rowTy = RankedTensorType::get({static_cast<int64_t>(row.size())},
                                       b.getF32Type());
    Value table = stablehlo::ConstantOp::create(
        b, loc, rowTy, DenseElementsAttr::get(rowTy, row));
    SmallVector<int64_t> idxShape(fTy.getShape());
    idxShape.push_back(1);
    Value indices = stablehlo::ReshapeOp::create(
        b, loc, RankedTensorType::get(idxShape, b.getI32Type()), idx);
    auto dnums = stablehlo::GatherDimensionNumbersAttr::get(
        b.getContext(), /*offsetDims=*/{}, /*collapsedSliceDims=*/{0},
        /*operandBatchingDims=*/{}, /*startIndicesBatchingDims=*/{},
        /*startIndexMap=*/{0}, /*indexVectorDim=*/fTy.getRank());
    return stablehlo::GatherOp::create(b, loc, fTy, table, indices, dnums,
                                       ArrayRef<int64_t>{1},
                                       /*indices_are_sorted=*/false);
  }

  // Splits x into (hi, lo) with 12 mantissa bits in hi, so that a product
  // of two hi parts is exact in float32.
  std::pair<Value, Value> split12(Value x) {
    Value hi = fromBits(land(toBits(x), i(-0x1000)));
    return {hi, sub(x, hi)};
  }

  // ln(x * 2^-subShift) as a compensated (hi, lo) pair.
  std::pair<Value, Value> logF32(Value x, Value subShift) {
    Value bits = toBits(x);
    Value expOff = sra(sub(bits, i(kSqrt2Over2Bits)), i(23));
    Value k = toF(subShift ? sub(expOff, subShift) : expOff);
    Value m = fromBits(sub(bits, shl(expOff, i(23))));
    Value fr = sub(m, f(1.0f));
    Value f2 = mul(fr, fr);
    Value ln1pf =
        add(sub(fr, mul(f(0.5f), f2)), mul(mul(f2, fr), horner(fr, kLogQHi)));
    Value hD = mul(k, f(kLn2Hi));
    Value lD = add(ln1pf, mul(k, f(kLn2Lo)));
    Value logHi = add(hD, lD);
    Value logLo = add(sub(hD, logHi), lD);
    return {logHi, logLo};
  }

  Value lgamma(Value x);
  Value digamma(Value x);

private:
  OpBuilder &b;
  Location loc;
  RankedTensorType fTy;
  RankedTensorType iTy;
};

Value Builder::lgamma(Value x) {
  // Normalize subnormals with integer arithmetic, so the result stays exact
  // on hardware that flushes subnormals to zero.
  Value bits = toBits(x);
  Value absBits = land(bits, i(0x7FFFFFFF));
  Value isZero = eq(absBits, i(0));
  Value isSub = land(gt(absBits, i(0)), lt(absBits, i(0x00800000)));
  Value subShift = select(isSub, sub(clz(absBits), i(8)), i(0));
  Value xNorm =
      fromBits(select(isSub,
                      lor(land(bits, i(std::numeric_limits<int32_t>::min())),
                          shl(absBits, subShift)),
                      bits));

  Value isPos = ge(xNorm, f(0.5f));
  Value notPos = lnot(isPos);
  // 1.0 for 0.0 keeps the unused branches free of 0/0 and log(0).
  Value xSafe = select(isZero, f(1.0f), xNorm);
  // Below -2^23 every float32 is an integer pole. The clamp to -0.5 keeps
  // round() finite and delta away from zero in the unused branches.
  Value xNegSafe =
      select(land(notPos, gt(xSafe, f(kAllIntegersBelow))), xSafe, f(-0.5f));
  // x = n + delta with |delta| <= 0.5, exact by Sterbenz. y = 1 - x as a
  // Fast2Sum pair, because 1 - n can dwarf delta.
  Value n = roundEven(xNegSafe);
  Value delta = sub(xNegSafe, n);
  Value absDelta = abs(delta);
  Value oneMinusN = sub(f(1.0f), n);
  Value y = sub(oneMinusN, delta);
  Value yLo = sub(sub(oneMinusN, y), delta);

  // The clamp prevents inf - inf in the Stirling series.
  Value xb = min(select(isPos, xSafe, y), f(kMaxFiniteArg));

  auto [logHi, logLo] = logF32(absDelta, subShift);

  // lgamma on [0.5, 4]: (x - 1)(x - 2) P(d), factors split hi/lo so the
  // unsummed pair keeps extra bits for the reflection.
  Value in154 = gt(xb, f(1.5f));
  Value u = sub(xb, f(1.0f));
  Value v = sub(u, f(1.0f));
  Value d = sub(u, select(in154, f(1.75f), f(0.0f)));
  SmallVector<Value> coeffs;
  for (auto [c1, c2] : llvm::zip_equal(kPosHi0515, kPosHi154))
    coeffs.push_back(select(in154, f(c2), f(c1)));
  Value p = horner(d, coeffs);
  auto [uHi, uLo] = split12(u);
  auto [vHi, vLo] = split12(v);
  Value uvHi = mul(uHi, vHi);
  Value uvLo = add(mul(uLo, v), mul(uHi, vLo));
  auto [pHi, pLo] = split12(p);
  Value smallHi = mul(uvHi, pHi);
  Value smallLo = add(mul(uvLo, p), mul(uvHi, pLo));
  Value small = add(smallHi, smallLo);

  // lgamma above 4: (x - 1/2)(ln x - 1) + (ln(2 pi) - 1)/2 + S(1/x^2)/x,
  // with an exact 12-bit leading product.
  auto [s, sLogLo] = logF32(xb, Value());
  auto [hX, sLo] = split12(s);
  Value lX = add(sLo, sLogLo);
  Value bb = sub(xb, f(0.5f));
  auto [bHi, bLo] = split12(bb);
  Value hxM1 = sub(hX, f(1.0f));
  Value largeHi = mul(bHi, hxM1);
  Value largeLoP = add(mul(bLo, hxM1), mul(bb, lX));
  Value invX = div(f(1.0f), xb);
  Value sp = horner(mul(invX, invX), kStirlingHi);
  Value largeLo = add(add(largeLoP, f(kHalfLn2piMinusHalf)), mul(invX, sp));
  Value psi = sub(s, mul(f(0.5f), invX));

  Value inLe4 = le(xb, f(4.0f));
  Value lgPos = select(inLe4, small, add(largeHi, largeLo));
  Value lgPosHi = select(inLe4, smallHi, largeHi);
  Value lgPosLo = select(inLe4, smallLo, largeLo);

  // Euler reflection with the first-order correction psi(y) * yLo.
  Value d2 = mul(delta, delta);
  Value sinc = mul(d2, horner(d2, kSincHi));
  Value refl = neg(add(add(logHi, lgPosHi),
                       add(add(lgPosLo, logLo), add(sinc, mul(psi, yLo)))));

  // Root windows near the negative zeros on (-10, -2). On (-2.6, -2) the
  // window belongs to the zero near -2.457, so n = -2 there.
  Value nRoot =
      select(land(gt(xNegSafe, f(-2.6f)), lt(xNegSafe, f(-2.0f))), f(-2.0f), n);
  Value deltaRoot = sub(xNegSafe, nRoot);
  Value k = toI(neg(nRoot));
  Value inK = land(ge(k, i(2)), le(k, i(9)));
  Value side = select(gt(deltaRoot, f(0.0f)), i(1), i(0));
  Value idx = select(inK, add(mul(i(2), sub(k, i(2))), side), i(1));
  SmallVector<Value> cols;
  for (const auto &row : kNegTable)
    cols.push_back(lookup(row, idx));
  Value rHi = cols[0], rLo = cols[1], wBound = cols[2];
  Value d0 = sub(deltaRoot, rHi);
  Value dr = add(deltaRoot, rHi);
  Value rcp = div(f(1.0f), dr);
  Value w = mul(sub(d0, rLo), rcp);
  w = add(w, mul(sub(sub(d0, mul(w, dr)), mul(rLo, add(f(1.0f), w))), rcp));
  Value inWin = le(abs(w), wBound);
  Value wSafe = select(inWin, w, f(0.0f));
  Value rw = mul(wSafe, horner(wSafe, ArrayRef<Value>(cols).drop_front(3)));

  Value res = select(isPos, lgPos, select(inWin, rw, refl));
  // +0.0, not -0.0, at x = 1 and x = 2.
  res = select(eq(res, f(0.0f)), f(0.0f), res);
  // +inf at the poles (0, negative integers, -inf) and on overflow.
  Value isNegInt = land(notPos, eq(xSafe, floor(xSafe)));
  Value isPoleOrInf = lor(lor(isZero, isNegInt), gt(x, f(kMaxFiniteArg)));
  res = select(isPoleOrInf, f(std::numeric_limits<float>::infinity()), res);
  return select(ne(x, x), f(std::numeric_limits<float>::quiet_NaN()), res);
}

Value Builder::digamma(Value x) {
  Value one = f(1.0f);
  Value isRefl = lt(x, f(0.5f));
  // pi cot(pi x) through x - round(x), which is exact and keeps the sine
  // argument small.
  Value dx = sub(x, roundEven(x));
  Value pd = mul(f(kPi), dx);
  Value cot = div(mul(f(kPi), cos(pd)), sin(pd));
  Value xr = select(isRefl, sub(one, x), x);

  // z >= 6: asymptotic series.
  Value isBig = ge(xr, f(6.0f));
  Value z = select(isBig, xr, f(6.0f));
  Value iz = div(one, z);
  Value iz2 = mul(iz, iz);
  Value asym =
      sub(sub(log(z), mul(f(0.5f), iz)), mul(iz2, horner(iz2, kDigammaTail)));

  // 0.5 <= z < 6: recurrence to [1, 2], adding the smallest terms first.
  Value y = select(isBig, f(1.5f), xr);
  Value acc = f(0.0f);
  for (int step = 0; step < 4; ++step) {
    Value down = gt(y, f(2.0f));
    y = select(down, sub(y, one), y);
    acc = select(down, add(acc, div(one, y)), acc);
  }
  Value up = lt(y, one);
  acc = select(up, sub(acc, div(one, y)), acc);
  y = select(up, add(y, one), y);
  // y - kDigammaRootHi is exact by Sterbenz on [1, 2].
  Value core = mul(sub(sub(y, f(kDigammaRootHi)), f(kDigammaRootLo)),
                   horner(sub(y, one), kDigammaCore));
  Value pos = select(isBig, asym, add(core, acc));

  Value res = select(isRefl, sub(pos, cot), pos);
  // NaN at the negative integers and -inf, -inf at +-0, as XLA returns.
  Value isNonPosInt = land(le(x, f(0.0f)), eq(x, floor(x)));
  res = select(isNonPosInt, f(std::numeric_limits<float>::quiet_NaN()), res);
  return select(eq(x, f(0.0f)), f(-std::numeric_limits<float>::infinity()),
                res);
}

bool isStaticF32(Type ty) {
  auto tt = dyn_cast<RankedTensorType>(ty);
  return tt && tt.hasStaticShape() && tt.getElementType().isF32();
}

struct LowerChloGammaF32Pass
    : public enzyme::impl::LowerChloGammaF32PassBase<LowerChloGammaF32Pass> {
  using LowerChloGammaF32PassBase::LowerChloGammaF32PassBase;

  void runOnOperation() override {
    SmallVector<Operation *> ops;
    getOperation()->walk([&](Operation *op) {
      if (isa<chlo::LgammaOp, chlo::DigammaOp>(op) &&
          isStaticF32(op->getResult(0).getType()))
        ops.push_back(op);
    });
    for (Operation *op : ops) {
      OpBuilder b(op);
      Builder e(b, op->getLoc(),
                cast<RankedTensorType>(op->getResult(0).getType()));
      Value x = op->getOperand(0);
      Value r = isa<chlo::LgammaOp>(op) ? e.lgamma(x) : e.digamma(x);
      op->getResult(0).replaceAllUsesWith(r);
      op->erase();
    }
  }
};

} // namespace
