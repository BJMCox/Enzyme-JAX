"""Tensorize imported HLO with Enzyme's host-independent batching pass."""

from functools import lru_cache

from jax.interpreters import mlir
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import func

from . import enzyme_call
from ._hlo_linearize import _external_values, _pure, _tensor, _walk

# Unknown operations retain pointwise execution. In particular, batching an RNG
# state is not equivalent to asking one RNG state for a larger output tensor.
_ELEMENTWISE = frozenset(
    "abs add and atan2 bitcast_convert cbrt ceil compare complex convert cosine "
    "divide exponential exponential_minus_one floor imag is_finite log log_plus_one "
    "logistic maximum minimum multiply negate not or popcnt power real remainder "
    "round_nearest_afz round_nearest_even rsqrt select shift_left shift_right_arithmetic "
    "shift_right_logical sign sine sqrt subtract tan tanh xor".split()
)
_SHAPED = frozenset(
    "broadcast_in_dim concatenate constant dot_general dynamic_slice "
    "dynamic_update_slice gather get_dimension_size iota optimization_barrier "
    "pad reshape reverse slice transpose".split()
)
# CHLO unary operations preserve each lane's shape. Keep their native AD rules
# instead of differentiating an expanded special-function approximation.
_CHLO_UNARY = frozenset(
    "_asin_acos_kernel acos acosh asin asinh atan atanh bessel_i1e conj cosh "
    "digamma erf erf_inv erfc is_inf is_neg_inf is_pos_inf lgamma sinh square tan".split()
)


def _batchable(op):
    name = op.name.removeprefix("stablehlo.")
    if op.name == "func.return" or name == "return":
        return True
    if not _pure(op):
        return False
    if not all(_tensor(v.type) for v in (*op.operands, *op.results)):
        return False
    if op.name.startswith("chlo."):
        return op.name.removeprefix("chlo.") in _CHLO_UNARY and not op.regions
    if not op.name.startswith("stablehlo."):
        return False
    if name in _ELEMENTWISE or name in _SHAPED:
        return not op.regions
    if name in ("reduce", "reduce_window", "scatter"):
        if name != "scatter":
            count = len(op.results)
            if not all(
                isinstance(v, ir.OpResult)
                and v.owner.operation.name == "stablehlo.constant"
                for v in list(op.operands)[count:]
            ):
                return False
        for region in op.regions:
            for block in region.blocks:
                local = set(block.arguments)
                for child in block.operations:
                    if not _external_values(child.operation) <= local:
                        return False
                    local.update(child.results)
    elif name != "while":
        return False
    return all(
        _batchable(child.operation)
        for region in op.regions
        for block in region.blocks
        for child in block.operations
    )


@lru_cache(maxsize=32)
def batch_hlo(source, pipeline, size, post_pipeline=""):
    """Return a tensorized module, or None for the pointwise fallback."""
    try:
        return _batch_hlo(source, pipeline, size, post_pipeline)
    except (ValueError, RuntimeError, ir.MLIRError):
        # Batching support must not narrow the imported program's scalar ABI.
        return None


def _batch_hlo(source, pipeline, size, post_pipeline):
    if not size:
        return None
    passes = pipeline + "," if pipeline else ""
    name, lowered = enzyme_call.run_pass_pipeline(
        [], source, passes + "tensor-empty-raise,drop-unsupported-attributes,inline"
    )
    with mlir.make_ir_context() as context, ir.Location.unknown():
        context.allow_unregistered_dialects = True
        module = ir.Module.parse(lowered)
        original = next(
            f
            for f in module.body.operations
            if f.operation.name == "func.func" and f.name.value == name
        )
        if not all(_batchable(op.operation) for op in original.entry_block.operations):
            return None
        inputs, outputs = list(original.type.inputs), list(original.type.results)
        if not all(_tensor(ty) for ty in inputs + outputs):
            return None
        names = {f.attributes["sym_name"].value for f in module.body.operations}
        cell = "enzyme_batch_cell"
        while cell in names:
            cell += "_"
        ir.SymbolTable.replace_all_symbol_uses(name, cell, module.operation)
        original.attributes["sym_name"] = ir.StringAttr.get(cell)
        original.attributes["sym_visibility"] = ir.StringAttr.get("private")

        def batched(ty):
            return ir.RankedTensorType.get((size, *ty.shape), ty.element_type)

        with ir.InsertionPoint(module.body):
            wrapper = func.FuncOp(
                "main", ([batched(t) for t in inputs], [batched(t) for t in outputs])
            )
            with ir.InsertionPoint(wrapper.add_entry_block()):
                call = ir.Operation.create(
                    "enzyme.batch",
                    results=[batched(t) for t in outputs],
                    operands=list(wrapper.entry_block.arguments),
                    attributes={
                        "fn": ir.FlatSymbolRefAttr.get(cell),
                        "batch_shape": ir.DenseI64ArrayAttr.get([size]),
                    },
                )
                func.ReturnOp(call.results)
        batched_source = str(module)
    name, result = enzyme_call.run_pass_pipeline(
        [],
        batched_source,
        "enzyme-batch,arith-raise{stablehlo=true},inline,canonicalize,cse,symbol-dce"
        + ",tensor-empty-raise,drop-unsupported-attributes",
    )
    with mlir.make_ir_context(), ir.Location.unknown():
        module = ir.Module.parse(result)
        function = next(f for f in module.body.operations if f.name.value == name)
        # This module is a new source program, which a later AD pipeline may
        # optimize again. Its entry must survive symbol dead-code elimination.
        function.attributes["sym_visibility"] = ir.StringAttr.get("public")
        # The optimizer's loop range analysis does not preserve unsigned
        # comparisons or narrowing counter updates. Optimize tensor graphs
        # here; leave native scans and per-operation loop fallbacks intact.
        if post_pipeline and not any(
            op.name == "stablehlo.while" for op in _walk(function.operation)
        ):
            name, result = enzyme_call.run_pass_pipeline(
                [],
                str(module),
                post_pipeline + ",tensor-empty-raise,drop-unsupported-attributes",
            )
            module = ir.Module.parse(result)
            function = next(f for f in module.body.operations if f.name.value == name)
            function.attributes["sym_visibility"] = ir.StringAttr.get("public")
        return str(module)
