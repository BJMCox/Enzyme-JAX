"""Share primal work by partitioning Enzyme's joint value/VJP into tensor residuals."""

from functools import lru_cache

from jax.interpreters import mlir
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import func

from . import enzyme_call


def _walk(op):
    yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _walk(child.operation)


def _external_values(op):
    # Region operands also include values captured without an explicit operand.
    internal = set()
    for child in _walk(op):
        internal.update(child.results)
        for region in child.regions:
            for block in region.blocks:
                internal.update(block.arguments)
    return {v for child in _walk(op) for v in child.operands} - internal


def _pure(op):
    try:
        if ir.MemoryEffectsOpInterface(op).get_effects():
            return False
    except ValueError:
        if not op.has_trait(ir.RecursiveMemoryEffectsTrait):
            return False
    return all(
        _pure(child.operation)
        for region in op.regions
        for block in region.blocks
        for child in block.operations
    )


def _tensor(ty):
    return isinstance(ty, ir.RankedTensorType) and ty.has_static_shape


def _extract(arguments, operations, results):
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        function = func.FuncOp(
            "main", ([v.type for v in arguments], [v.type for v in results])
        )
        block = function.add_entry_block()
        mapped = dict(zip(arguments, block.arguments))
        with ir.InsertionPoint(block):
            for op in operations:
                clone = op.operation.clone()
                for child in _walk(clone):
                    for index, operand in enumerate(child.operands):
                        if operand in mapped:
                            child.operands[index] = mapped[operand]
                mapped.update(zip(op.results, clone.results))
            func.ReturnOp([mapped[v] for v in results])
    module.operation.verify()
    return str(module)


@lru_cache(maxsize=32)
def linearize_hlo(source, activity, before_ad, after_ad):
    """Return pure forward/reverse modules, or None to retain combined reverse AD.

    Enzyme computes every derivative. This partition only moves operations
    independent of output cotangents into the forward function, and retains
    their crossing SSA values for the reverse function.
    """
    if not (
        hasattr(getattr(ir, "MemoryEffectsOpInterface", None), "get_effects")
        and hasattr(ir.Operation, "has_trait")
        and hasattr(ir, "RecursiveMemoryEffectsTrait")
    ):
        return None
    try:
        name, optimized = enzyme_call.run_pass_pipeline([], source, before_ad)
        with mlir.make_ir_context() as context, ir.Location.unknown():
            context.allow_unregistered_dialects = True
            module = ir.Module.parse(optimized)
            original = next(
                f
                for f in module.body.operations
                if f.operation.name == "func.func" and f.name.value == name
            )
            inputs, outputs = list(original.type.inputs), list(original.type.results)
            if (
                len(inputs) != len(activity)
                or len(original.regions[0].blocks) != 1
                or not all(_tensor(ty) for ty in inputs + outputs)
                or not all(str(ty.element_type) in ("f32", "f64") for ty in outputs)
                or not all(
                    _pure(op.operation) for op in original.entry_block.operations
                )
            ):
                return None
            names = {
                f.attributes["sym_name"].value
                for f in module.body.operations
                if "sym_name" in f.attributes
            }
            primal_name = "enzyme_primal"
            while primal_name in names:
                primal_name += "_"
            ir.SymbolTable.replace_all_symbol_uses(name, primal_name, module.operation)
            original.attributes["sym_name"] = ir.StringAttr.get(primal_name)
            original.attributes["sym_visibility"] = ir.StringAttr.get("private")
            gradients = [ty for ty, active in zip(inputs, activity) if active]
            if not all(str(ty.element_type) in ("f32", "f64") for ty in gradients):
                return None
            with ir.InsertionPoint(module.body):
                wrapper = func.FuncOp("main", (inputs + outputs, outputs + gradients))
                with ir.InsertionPoint(wrapper.add_entry_block()):
                    active = ir.Attribute.parse("#enzyme<activity enzyme_active>")
                    constant = ir.Attribute.parse("#enzyme<activity enzyme_const>")
                    call = ir.Operation.create(
                        "enzyme.autodiff",
                        results=outputs + gradients,
                        operands=list(wrapper.entry_block.arguments),
                        attributes={
                            "fn": ir.FlatSymbolRefAttr.get(primal_name),
                            "activity": ir.ArrayAttr.get(
                                [active if a else constant for a in activity]
                            ),
                            "ret_activity": ir.ArrayAttr.get([active] * len(outputs)),
                        },
                    )
                    func.ReturnOp(call.results)
            joint = str(module)
        name, joint = enzyme_call.run_pass_pipeline(
            [],
            joint,
            "enzyme,arith-raise{stablehlo=true},enzyme-batch-to-stablehlo,"
            "canonicalize,remove-unnecessary-enzyme-ops,"
            + after_ad
            + ",tensor-empty-raise,drop-unsupported-attributes",
        )
        return _partition(joint, name, len(inputs), len(outputs))
    except (ValueError, RuntimeError, ir.MLIRError):
        # The transpose may later mark outputs inactive. Do not make a working
        # activity-specific derivative depend on speculative all-active AD.
        return None


def _partition(joint, name, input_count, output_count):
    with mlir.make_ir_context(), ir.Location.unknown():
        module = ir.Module.parse(joint)
        # Extraction must remain self-contained. Retain the established path
        # when inlining leaves another function or module-level symbol.
        if len(module.body.operations) != 1:
            return None
        function = module.body.operations[0]
        if function.name.value != name or len(function.regions[0].blocks) != 1:
            return None
        block = function.entry_block
        if not all(_pure(op.operation) for op in block.operations):
            return None
        arguments = list(block.arguments)
        known = set(arguments[:input_count])
        forward, reverse = [], []
        for op in list(block.operations)[:-1]:
            if _external_values(op.operation) <= known:
                forward.append(op)
                known.update(op.results)
            else:
                reverse.append(op)
        returned = list(block.operations[-1].operands)
        values, gradients = returned[:output_count], returned[output_count:]
        if not set(values) <= known:
            return None
        used = set(gradients)
        for op in reverse:
            used.update(_external_values(op.operation))
        ordered = arguments[:input_count] + [v for op in forward for v in op.results]
        residuals = [v for v in ordered if v in used]
        if not all(_tensor(v.type) for v in residuals):
            return None
        return (
            _extract(arguments[:input_count], forward, values + residuals),
            _extract(residuals + arguments[input_count:], reverse, gradients),
        )
