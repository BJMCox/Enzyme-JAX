"""Preserve versioned portable linear algebra for implicit differentiation.

Reserved private functions return (solve(A, B), valid), with zero output on
invalid cells. Their bodies guard invalid operations. The name asserts this
contract; signature and purity checks alone cannot prove it.
Logabsdet functions return (logabsdet(A), inverse(A), valid); their inverse is
undefined on invalid cells, as is the log determinant's derivative there.
"""

from jax.interpreters import mlir
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import func

from ._hlo_linearize import _pure, _tensor, _walk

_PREFIX = "__enzyme_linalg_solve_v1_"
_LOGABSDET_PREFIX = "__enzyme_linalg_logabsdet_v1_"
_NAME = "enzyme.linalg.solve"
_FACTOR = "__enzyme_linalg_factor_v1"
_CACHED_PREFIX = "__enzyme_linalg_cached_solve_v2_"


def cache_solve_factors(module, function):
    """Expose integer implementation caches before AD partitions the solve.

    These executor-internal calls require target lowering before execution.
    The original matrix remains an operand of each solve and owns its implicit
    derivatives; integer LU storage must never detach that matrix dependence.
    """
    definitions = {
        f.name.value: f
        for f in module.body.operations
        if f.operation.name == "func.func"
    }
    wrappers = {}

    def rewrite(block):
        factors = {}
        for view in list(block.operations):
            op = view.operation
            for region in op.regions:
                for child in region.blocks:
                    rewrite(child)
            if (
                op.name != "stablehlo.composite"
                or ir.StringAttr(op.attributes["name"]).value != _NAME
                or "version" not in op.attributes
                or ir.IntegerAttr(op.attributes["version"]).value != 1
                or (
                    "composite_attributes" in op.attributes
                    and len(ir.DictAttr(op.attributes["composite_attributes"]))
                )
            ):
                continue
            decomposition = ir.FlatSymbolRefAttr(op.attributes["decomposition"]).value
            if not decomposition.startswith(_PREFIX):
                continue
            try:
                _check_solve(op, decomposition, definitions)
            except ValueError:
                continue
            a, b = op.operands
            a_type, b_type = (ir.RankedTensorType(v.type) for v in (a, b))
            if b_type.shape[-1] == 0 or 0 in a_type.shape[:-2]:
                continue
            original, transpose = a, False
            while isinstance(original, ir.OpResult):
                trans = original.owner.operation
                shape = ir.RankedTensorType(original.type).shape
                rank = len(shape)
                if trans.name == "stablehlo.transpose" and list(
                    ir.DenseI64ArrayAttr(trans.attributes["permutation"])
                ) == [*range(rank - 2), rank - 1, rank - 2]:
                    original = trans.operands[0]
                    transpose = not transpose
                    continue
                if trans.name == "stablehlo.broadcast_in_dim":
                    inner = ir.RankedTensorType(trans.operands[0].type)
                    added = rank - len(inner.shape)
                    if (
                        added > 0
                        and len(inner.shape) >= 2
                        and tuple(shape[added:]) == tuple(inner.shape)
                        and list(
                            ir.DenseI64ArrayAttr(
                                trans.attributes["broadcast_dimensions"]
                            )
                        )
                        == list(range(added, rank))
                    ):
                        original = trans.operands[0]
                        continue
                break
            ty = ir.RankedTensorType(a.type)
            bits = ir.IntegerType.get_signless(
                32 if str(ty.element_type) == "f32" else 64
            )
            cache_types = [
                ir.RankedTensorType.get(ty.shape, bits),
                ir.RankedTensorType.get(ty.shape[:-1], ir.IntegerType.get_signless(32)),
                op.results[1].type,
            ]
            with ir.InsertionPoint(op), op.location:
                if original not in factors:
                    inner = ir.RankedTensorType(original.type)
                    factor_types = [
                        ir.RankedTensorType.get(inner.shape, bits),
                        ir.RankedTensorType.get(
                            inner.shape[:-1], ir.IntegerType.get_signless(32)
                        ),
                        ir.RankedTensorType.get(
                            inner.shape[:-2], ir.IntegerType.get_signless(1)
                        ),
                    ]
                    factor = ir.Operation.create(
                        "stablehlo.custom_call",
                        operands=[original],
                        results=factor_types,
                        attributes={
                            "call_target_name": ir.StringAttr.get(_FACTOR),
                            "has_side_effect": ir.BoolAttr.get(False),
                        },
                    )
                    factors[original] = list(factor.results)
                cache = []
                for value, target in zip(factors[original], cache_types):
                    if value.type != target:
                        inner_rank = len(ir.RankedTensorType(value.type).shape)
                        outer_rank = len(ir.RankedTensorType(target).shape)
                        value = ir.Operation.create(
                            "stablehlo.broadcast_in_dim",
                            operands=[value],
                            results=[target],
                            attributes={
                                "broadcast_dimensions": ir.DenseI64ArrayAttr.get(
                                    list(range(outer_rank - inner_rank, outer_rank))
                                )
                            },
                        ).result
                    cache.append(value)
                flag = ir.Operation.create(
                    "stablehlo.constant",
                    results=[op.results[1].type],
                    attributes={
                        "value": ir.DenseElementsAttr.get_splat(
                            op.results[1].type, ir.BoolAttr.get(transpose)
                        )
                    },
                ).result
                operands = [a, b, *cache, flag]
                if decomposition not in wrappers:
                    name = _CACHED_PREFIX + decomposition
                    while name in definitions:
                        name += "_"
                    with ir.InsertionPoint(module.body):
                        fallback = func.FuncOp(
                            name,
                            ([v.type for v in operands], [v.type for v in op.results]),
                            visibility="private",
                        )
                        with ir.InsertionPoint(fallback.add_entry_block()):
                            result = func.CallOp(
                                [v.type for v in op.results],
                                decomposition,
                                list(fallback.entry_block.arguments[:2]),
                            )
                            func.ReturnOp(
                                [result.results[0], fallback.entry_block.arguments[4]]
                            )
                    definitions[name] = fallback
                    wrappers[decomposition] = name
                name = wrappers[decomposition]
                cached = ir.Operation.create(
                    "stablehlo.composite",
                    operands=operands,
                    results=[v.type for v in op.results],
                    attributes={
                        "name": ir.StringAttr.get(_NAME),
                        "decomposition": ir.FlatSymbolRefAttr.get(name),
                        "composite_attributes": ir.DictAttr.get({}),
                        "version": ir.IntegerAttr.get(
                            ir.IntegerType.get_signless(32), 2
                        ),
                    },
                )
            for old, new in zip(op.results, cached.results):
                old.replace_all_uses_with(new)
            op.erase()

    rewrite(function.entry_block)


def pure_function(function, functions, active=None):
    active = set() if active is None else active
    name = function.name.value
    if name in active:
        return False
    if len(function.regions[0].blocks) != 1:
        return False
    active = active | {name}

    return all(
        pure_operation(view.operation, functions, active)
        for view in function.entry_block.operations
    )


def pure_operation(op, functions, active=None):
    if op.name in ("func.call", "stablehlo.composite"):
        key = "callee" if op.name == "func.call" else "decomposition"
        callee = ir.FlatSymbolRefAttr(op.attributes[key]).value
        target = functions.get(callee)
        return target is not None and pure_function(target, functions, active)
    return _pure(op, lambda child: pure_operation(child, functions, active))


def _check_solve(call, name, functions):
    function = functions.get(name)
    if (
        function is None
        or "sym_visibility" not in function.attributes
        or ir.StringAttr(function.attributes["sym_visibility"]).value != "private"
        or len(call.operands) != 2
        or len(call.results) != 2
        or list(function.type.inputs) != [v.type for v in call.operands]
        or list(function.type.results) != [v.type for v in call.results]
        or not pure_function(function, functions)
        or not all(_tensor(v.type) for v in (*call.operands, *call.results))
    ):
        raise ValueError("invalid reserved solve function contract")
    a, b = (ir.RankedTensorType(v.type) for v in call.operands)
    x, valid = (ir.RankedTensorType(v.type) for v in call.results)
    if (
        len(a.shape) < 2
        or len(a.shape) != len(b.shape)
        or tuple(a.shape[:-2]) != tuple(b.shape[:-2])
        or tuple(valid.shape) != tuple(a.shape[:-2])
        or a.shape[-1] == 0
        or a.shape[-1] != a.shape[-2]
        or b.shape[-2] != a.shape[-1]
        or str(a.element_type) not in ("f32", "f64")
        or a.element_type != b.element_type
        or x != b
        or str(valid.element_type) != "i1"
    ):
        raise ValueError("invalid reserved solve tensor signature")


def _check_logabsdet(call, name, functions):
    function = functions.get(name)
    if (
        function is None
        or "sym_visibility" not in function.attributes
        or ir.StringAttr(function.attributes["sym_visibility"]).value != "private"
        or len(call.operands) != 1
        or len(call.results) != 3
        or list(function.type.inputs) != [v.type for v in call.operands]
        or list(function.type.results) != [v.type for v in call.results]
        or not pure_function(function, functions)
        or not all(_tensor(v.type) for v in (*call.operands, *call.results))
    ):
        raise ValueError("invalid reserved logabsdet function contract")
    a = ir.RankedTensorType(call.operands[0].type)
    value, inverse, valid = (ir.RankedTensorType(v.type) for v in call.results)
    if (
        len(a.shape) < 2
        or a.shape[-1] == 0
        or a.shape[-1] != a.shape[-2]
        or str(a.element_type) not in ("f32", "f64")
        or inverse != a
        or tuple(value.shape) != tuple(a.shape[:-2])
        or value.element_type != a.element_type
        or tuple(valid.shape) != tuple(a.shape[:-2])
        or str(valid.element_type) != "i1"
    ):
        raise ValueError("invalid reserved logabsdet tensor signature")


def recognize_linalg(source):
    if _PREFIX not in source and _LOGABSDET_PREFIX not in source:
        return source
    with mlir.make_ir_context(), ir.Location.unknown():
        module = ir.Module.parse(source)
        functions = {
            f.name.value: f
            for f in module.body.operations
            if f.operation.name == "func.func"
        }
        calls = [
            op
            for op in _walk(module.operation)
            if op.name == "func.call"
            and ir.FlatSymbolRefAttr(op.attributes["callee"]).value.startswith(
                (_PREFIX, _LOGABSDET_PREFIX)
            )
        ]
        for call in calls:
            name = ir.FlatSymbolRefAttr(call.attributes["callee"]).value
            logarithm = name.startswith(_LOGABSDET_PREFIX)
            (_check_logabsdet if logarithm else _check_solve)(call, name, functions)
            with ir.InsertionPoint(call):
                composite = ir.Operation.create(
                    "stablehlo.composite",
                    operands=list(call.operands),
                    results=[v.type for v in call.results],
                    attributes={
                        "name": ir.StringAttr.get(
                            "enzyme.linalg.logabsdet" if logarithm else _NAME
                        ),
                        "decomposition": ir.FlatSymbolRefAttr.get(name),
                        "composite_attributes": ir.DictAttr.get({}),
                        "version": ir.IntegerAttr.get(
                            ir.IntegerType.get_signless(32), 1
                        ),
                    },
                )
            for old, new in zip(call.results, composite.results):
                old.replace_all_uses_with(new)
            call.erase()
        module.operation.verify()
        return str(module)
