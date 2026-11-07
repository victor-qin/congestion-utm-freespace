"""Matching optional Metal/CUDA entry points for checked integer pruning."""

from pathlib import Path


EXTRA_ARGUMENTS = (
    "GLOBAL const int* numeric_meta", "GLOBAL const unsigned int* numeric_roots",
    "GLOBAL const unsigned int* numeric_costs", "GLOBAL const int* numeric_prices",
    "GLOBAL const unsigned int* numeric_price_values",
    "GLOBAL const unsigned int* numeric_endpoints", "GLOBAL const unsigned int* numeric_future",
    "GLOBAL const unsigned int* numeric_cutoffs",
)


def extend_source(source: str, backend: str, arguments: tuple[str, ...]) -> str:
    """Append optional integer expansion/root kernels with identical backend interfaces.

    Parameters
    ------------
    - source (str): complete ordinary pricing runtime source.
    - backend (str): explicit Metal or CUDA target.
    - arguments (tuple[str, ...]): existing expansion arguments in their original order.

    Return
    --------
    - output (str): source retaining ordinary kernels while replacing expansion's entry wrapper.
    """
    source = "#define PRICING_NUMERIC_WORDS 15\n" + source.replace("void expand(", "void fp32_expand(")
    for name in ("pricing_integer.h", "pricing_integer_bounds.h", "pricing_integer_search.h"):
        source += Path(__file__).with_name(name).read_text()
    all_arguments = arguments + EXTRA_ARGUMENTS
    names = ", ".join(arg.split()[-1] for arg in arguments)
    extras = ", ".join(arg.split()[-1] for arg in EXTRA_ARGUMENTS)
    root_arguments = ("GLOBAL const int* exponents", "GLOBAL const unsigned int* operands",
                      "GLOBAL unsigned int* output")
    for name, args, body in (
        ("expand", all_arguments,
         f"expand_one(tid, {names}); integer_expand_one(tid, {names}, {extras});"),
        ("integer_roots", root_arguments,
         "if (tid < (unsigned int)exponents[0]) ii_store(output + tid * 5, "
         "ib_minimum(ib_upper(operands + tid * 60, exponents[tid + 1]), "
         "ii_load(operands + tid * 60 + 55)));"),
    ):
        if backend == "metal":
            formatted = ", ".join(f"{arg} [[buffer({i})]]" for i, arg in enumerate(args))
            source += f"\nkernel void {name}({formatted}, uint tid [[thread_position_in_grid]]) {{ {body} }}\n"
        else:
            source += (f'\nextern "C" __global__ void {name}({", ".join(args)}) {{ '
                       f"unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x; {body} }}\n")
    return source
