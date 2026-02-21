from abc import ABC, abstractmethod
from dataclasses import dataclass

from lark import Lark, Tree


class EinsumExpr(ABC):
    @abstractmethod
    def get_loops(self) -> set[str]:
        pass

    @abstractmethod
    def run(self, xp, loops, kwargs):
        pass


nary_ops = {
    "+": "add",
    "add": "add",
    "-": "subtract",
    "sub": "subtract",
    "subtract": "subtract",
    "*": "multiply",
    "mul": "multiply",
    "multiply": "multiply",
    "/": "divide",
    "div": "divide",
    "divide": "divide",
    "//": "floor_divide",
    "fld": "floor_divide",
    "floor_divide": "floor_divide",
    "%": "remainder",
    "mod": "remainder",
    "remainder": "remainder",
    "**": "power",
    "pow": "power",
    "power": "power",
    "==": "equal",
    "eq": "equal",
    "equal": "equal",
    "!=": "not_equal",
    "ne": "not_equal",
    "not_equal": "not_equal",
    "<": "less",
    "lt": "less",
    "less": "less",
    "<=": "less_equal",
    "le": "less_equal",
    "less_equal": "less_equal",
    ">": "greater",
    "gt": "greater",
    "greater": "greater",
    ">=": "greater_equal",
    "ge": "greater_equal",
    "greater_equal": "greater_equal",
    "&": "bitwise_and",
    "bitwise_and": "bitwise_and",
    "|": "bitwise_or",
    "bitwise_or": "bitwise_or",
    "^": "bitwise_xor",
    "bitwise_xor": "bitwise_xor",
    "<<": "bitwise_left_shift",
    "lshift": "bitwise_left_shift",
    "bitwise_left_shift": "bitwise_left_shift",
    ">>": "bitwise_right_shift",
    "rshift": "bitwise_right_shift",
    "bitwise_right_shift": "bitwise_right_shift",
    "and": "logical_and",
    "or": "logical_or",
    "not": "logical_not",
    "min": "minimum",
    "max": "maximum",
    "logaddexp": "logaddexp",
}


unary_ops = {
    "+": "positive",
    "pos": "positive",
    "positive": "positive",
    "-": "negative",
    "neg": "negative",
    "negative": "negative",
    "~": "bitwise_invert",
    "invert": "bitwise_invert",
    "bitwise_invert": "bitwise_invert",
    "not": "logical_not",
    "logical_not": "logical_not",
    "abs": "absolute",
    "absolute": "absolute",
    "sqrt": "sqrt",
    "exp": "exp",
    "log": "log",
    "log1p": "log1p",
    "log10": "log10",
    "log2": "log2",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "asin": "arcsin",
    "acos": "arccos",
    "atan": "arctan",
    "asinh": "arcsinh",
    "acosh": "arccosh",
    "atanh": "arctanh",
}


reduction_ops = {
    "+": "sum",
    "add": "sum",
    "sum": "sum",
    "*": "prod",
    "mul": "prod",
    "prod": "prod",
    "and": "all",
    "or": "any",
    "min": "min",
    "max": "max",
    "argmin": "argmin",
    "argmax": "argmax",
    "mean": "mean",
    "std": "std",
    "var": "var",
    "count_nonzero": "count_nonzero",
    # "&": "bitwise_and",
    # "|": "bitwise_or",
    # "^": "bitwise_xor",
}


@dataclass
class Access(EinsumExpr):
    tns: str
    idxs: list[str]

    def get_loops(self) -> set[str]:
        return set(self.idxs)

    def run(self, xp, loops, kwargs):
        assert len(self.idxs) == len(set(self.idxs))
        perm = [self.idxs.index(idx) for idx in loops if idx in self.idxs]
        tns = kwargs[self.tns]
        tns = xp.transpose(tns, perm)
        return xp.expand_dims(
            tns, [i for i in range(len(loops)) if loops[i] not in self.idxs]
        )


@dataclass
class Literal(EinsumExpr):
    value: bool | int | float | complex

    def get_loops(self) -> set[str]:
        return set()

    def run(self, xp, loops, kwargs):
        # Create a scalar array with the same shape as needed
        shape = [1] * len(loops)
        return xp.full(shape, self.value)


@dataclass
class Call(EinsumExpr):
    func: str
    args: list[EinsumExpr]

    def get_loops(self) -> set[str]:
        return set().union(*[arg.get_loops() for arg in self.args])

    def run(self, xp, loops, kwargs):
        if len(self.args) == 1:
            func = getattr(xp, unary_ops[self.func])
        else:
            func = getattr(xp, nary_ops[self.func])
        vals = [arg.run(xp, loops, kwargs) for arg in self.args]
        return func(*vals)


@dataclass
class Einsum:
    arg: EinsumExpr
    op: str | None
    tns: str
    idxs: list[str]

    def run(self, xp, kwargs):
        # This is the main entry point for einsum execution
        loops = self.arg.get_loops()
        assert set(self.idxs).issubset(loops)
        loops = sorted(loops)
        arg = self.arg.run(xp, loops, kwargs)
        axis = tuple(i for i in range(len(loops)) if loops[i] not in self.idxs)
        if self.op is not None:
            op = getattr(xp, reduction_ops.get(self.op))
            val = op(arg, axis=axis)
        else:
            assert set(self.idxs) == set(loops)
            val = arg
        dropped = [idx for idx in loops if idx in self.idxs]
        axis = [dropped.index(idx) for idx in self.idxs]
        return xp.transpose(val, axis)


lark_parser = Lark("""
    %import common.CNAME
    %import common.SIGNED_INT
    %import common.SIGNED_FLOAT
    %ignore " "           // Disregard spaces in text

    start: increment | assign
    increment: access (OP | FUNC_NAME) "=" expr
    assign: access "=" expr

    // Python operator precedence (lowest to highest)
    expr: or_expr
    or_expr: and_expr (OR and_expr)*
    and_expr: not_expr (AND not_expr)*
    not_expr: NOT not_expr | comparison_expr
    comparison_expr: bitwise_or_expr ((EQ | NE | LT | LE | GT | GE) bitwise_or_expr)*
    bitwise_or_expr: bitwise_xor_expr (PIPE bitwise_xor_expr)*
    bitwise_xor_expr: bitwise_and_expr (CARET bitwise_and_expr)*
    bitwise_and_expr: shift_expr (AMPERSAND shift_expr)*
    shift_expr: add_expr ((LSHIFT | RSHIFT) add_expr)*
    add_expr: mul_expr ((PLUS | MINUS) mul_expr)*
    mul_expr: unary_expr ((MUL | DIV | FLOORDIV | MOD) unary_expr)*
    unary_expr: (PLUS | MINUS | TILDE) unary_expr | power_expr
    power_expr: primary (POW unary_expr)?
    primary: call_func | access | literal | "(" expr ")"

    OR: "or"
    AND: "and"
    NOT: "not"
    EQ: "=="
    NE: "!="
    LT: "<"
    LE: "<="
    GT: ">"
    GE: ">="
    PIPE: "|"
    CARET: "^"
    AMPERSAND: "&"
    LSHIFT: "<<"
    RSHIFT: ">>"
    PLUS: "+"
    MINUS: "-"
    MUL: "*"
    DIV: "/"
    FLOORDIV: "//"
    MOD: "%"
    POW: "**"
    TILDE: "~"

    OP: "+" | "-" | "*" | "or" | "and" | "|" | "&" | "^" | "<<" | ">>"
          | "//" | "/" | "%" | "**" | ">" | "<" | ">=" | "<=" | "==" | "!="

    access: TNS "[" (IDX ",")* IDX? "]"
    call_func: (FUNC_NAME "(" (expr ",")* expr?  ")")
    literal: bool_literal | complex_literal | float_literal | int_literal
    bool_literal: BOOL
    int_literal: SIGNED_INT
    float_literal: SIGNED_FLOAT
    complex_literal: COMPLEX

    BOOL: "True" | "False"
    COMPLEX: (SIGNED_FLOAT | SIGNED_INT) ("j" | "J")
    IDX: CNAME
    TNS: CNAME
    FUNC_NAME: CNAME
""")


def _parse_einsum_expr(t: Tree) -> EinsumExpr:
    match t:
        case Tree(
            "start"
            | "expr"
            | "or_expr"
            | "and_expr"
            | "not_expr"
            | "comparison_expr"
            | "bitwise_or_expr"
            | "bitwise_xor_expr"
            | "bitwise_and_expr"
            | "shift_expr"
            | "add_expr"
            | "mul_expr"
            | "unary_expr"
            | "power_expr"
            | "primary"
            | "literal",
            [child],
        ):
            return _parse_einsum_expr(child)
        case Tree(
            "or_expr"
            | "and_expr"
            | "bitwise_or_expr"
            | "bitwise_and_expr"
            | "bitwise_xor_expr"
            | "shift_expr"
            | "add_expr"
            | "mul_expr",
            args,
        ) if len(args) > 1:
            expr = _parse_einsum_expr(args[0])
            for i in range(1, len(args), 2):
                arg = _parse_einsum_expr(args[i + 1])
                expr = Call(args[i].value, [expr, arg])  # type: ignore[union-attr]
            return expr
        case Tree("comparison_expr", args) if len(args) > 1:
            # Handle Python's comparison chaining: a < b < c becomes (a < b) and (b < c)
            left = _parse_einsum_expr(args[0])
            right = _parse_einsum_expr(args[2])
            expr = Call(args[1].value, [left, right])  # type: ignore[union-attr]
            for i in range(2, len(args) - 2, 2):
                left = _parse_einsum_expr(args[i])
                right = _parse_einsum_expr(args[i + 2])
                expr = Call("and", [expr, Call(args[i + 1].value, [left, right])])  # type: ignore[union-attr]
            return expr
        case Tree("power_expr", args) if len(args) > 1:
            left = _parse_einsum_expr(args[0])
            right = _parse_einsum_expr(args[2])
            return Call(args[1].value, [left, right])  # type: ignore[union-attr]
        case Tree("unary_expr" | "not_expr", [op, arg]):
            return Call(op.value, [_parse_einsum_expr(arg)])  # type: ignore[union-attr]
        case Tree("access", [tns, *idxs]):
            return Access(tns.value, [idx.value for idx in idxs])  # type: ignore[union-attr]
        case Tree("bool_literal", [val]):
            return Literal(val.value == "True")  # type: ignore[union-attr]
        case Tree("int_literal", [val]):
            return Literal(int(val.value))  # type: ignore[union-attr]
        case Tree("float_literal", [val]):
            return Literal(float(val.value))  # type: ignore[union-attr]
        case Tree("complex_literal", [val]):
            return Literal(complex(val.value))  # type: ignore[union-attr]
        case Tree("call_func", [func, *args]):
            return Call(func.value, [_parse_einsum_expr(arg) for arg in args])  # type: ignore[union-attr]
        case _:
            raise ValueError(f"Unknown tree structure: {t}")


def parse_einsum(expr: str) -> Einsum:
    tree = lark_parser.parse(expr)
    print(f"Parsed tree: {tree.pretty()}")

    match tree:
        case Tree(
            "start", [Tree("increment", [Tree("access", [tns, *idxs]), op, expr_node])]
        ):
            input_expr = _parse_einsum_expr(expr_node)  # type: ignore[arg-type]
            return Einsum(input_expr, op.value, tns.value, [idx.value for idx in idxs])  # type: ignore[union-attr]

        case Tree("start", [Tree("assign", [Tree("access", [tns, *idxs]), expr_node])]):
            input_expr = _parse_einsum_expr(expr_node)  # type: ignore[arg-type]
            return Einsum(input_expr, None, tns.value, [idx.value for idx in idxs])  # type: ignore[union-attr]

        case _:
            raise ValueError(
                f"Expected top-level assignment or increment, got {tree.data}"
            )


def einsum(xp, prgm, **kwargs):
    """Execute an einsum expression using the specified array framework.

    This function parses and executes einsum-like expressions with extended syntax
    that supports various operations beyond traditional Einstein summation notation.

    Args:
        xp: Array framework module (e.g., numpy, cupy, or other array library)
            that provides the underlying array operations.
        prgm (str): Einsum program string specifying the computation. The syntax
            supports:
            - Assignment: "C[i,j] = A[i,j] + B[j,i]"
            - Increment: "C[i,j] += A[i,k] * B[k,j]"
            - Reductions: "C[i] += A[i,j]", "C[i] max= A[i,j]", "C[i] &= A[i,j]"
            - Arithmetic operations: +, -, *, /, //, %, **
            - Comparison operations: ==, !=, <, <=, >, >=
            - Logical operations: and, or, not
            - Bitwise operations: &, |, ^, <<, >>
            - Function calls and complex expressions with parentheses
            - Mathematical functions: abs, sqrt, exp, log, sin, cos, tan, etc.
            - Literal values: integers, floats, booleans, and complex numbers
            - Python operator precedence and parentheses for grouping
        **kwargs: Named arrays referenced in the einsum expression. The keys
            should match the tensor names used in the program string.

    Returns:
        The result array from executing the einsum expression.

    Examples:
        >>> import numpy as np
        >>> A = np.random.rand(3, 4)
        >>> B = np.random.rand(4, 3)
        >>> # Matrix addition with transpose
        >>> C = einsum(np, "C[i,j] = A[i,j] + B[j,i]", A=A, B=B)
        >>> # Matrix multiplication
        >>> D = einsum(np, "D[i,j] += A[i,k] * B[k,j]", A=A, B=B)
        >>> # Min-Plus multiplication with shift
        >>> E = einsum(np, "E[i] min= A[i,k] + D[k,j] << 1", A=A, D=D)
    """
    prgm = parse_einsum(prgm)
    return prgm.run(xp, kwargs)
