"""Preserve versioned portable solve calls for implicit differentiation.

Reserved private functions return (solve(A, B), valid), with zero output on
invalid cells. Their bodies guard invalid operations. The name asserts this
contract; signature and purity checks alone cannot prove it.
"""

from jax.interpreters import mlir
from jaxlib.mlir import ir

from ._hlo_linearize import _pure, _tensor, _walk

_PREFIX = "__enzyme_linalg_solve_v1_"
_NAME = "enzyme.linalg.solve"


def pure_function(function, functions, active=None):
    active = set() if active is None else active
    name = function.name.value
    if name in active:
        return False
    if len(function.regions[0].blocks) != 1:
        return False
    active = active | {name}
    for view in function.entry_block.operations:
        op = view.operation
        if op.name == "func.call":
            callee = ir.FlatSymbolRefAttr(op.attributes["callee"]).value
            target = functions.get(callee)
            if target is None or not pure_function(target, functions, active):
                return False
        elif not _pure(op):
            return False
    return True


def recognize_solve(source):
    if _PREFIX not in source:
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
            and ir.FlatSymbolRefAttr(op.attributes["callee"]).value.startswith(_PREFIX)
        ]
        for call in calls:
            name = ir.FlatSymbolRefAttr(call.attributes["callee"]).value
            function = functions.get(name)
            if (
                function is None
                or "sym_visibility" not in function.attributes
                or ir.StringAttr(function.attributes["sym_visibility"]).value
                != "private"
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
            with ir.InsertionPoint(call):
                composite = ir.Operation.create(
                    "stablehlo.composite",
                    operands=list(call.operands),
                    results=[v.type for v in call.results],
                    attributes={
                        "name": ir.StringAttr.get(_NAME),
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
