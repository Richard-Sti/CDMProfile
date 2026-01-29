# Copyright (C) 2024 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
Symbolic expression parser with C code generation.
"""
import numpy as np
from cffi import FFI
from sympy import (Abs, Lambda, Pow, Symbol, ccode, lambdify, log, sqrt,
                   symbols, sympify)

# Optional per-job CFFI cache directory (mirrors fitting.CFFI_TMPDIR).
CFFI_TMPDIR = None


class SympyParser:
    """
    SymPy parser for expressions of the form `f(x, a0, a1, ...)`.

    The variable `x` represents `r / Rs` (radius / scale radius) when
    `use_scaled_radius=True`, or simply `r` when `use_scaled_radius=False`.
    Parameters `a0, a1, ...` are free fitting parameters.

    Parameters
    ----------
    nfree_max : int, optional
        Maximum number of free parameters. Default is 4.
    use_scaled_radius : bool, optional
        If True (default), substitute x -> r/Rs. If False, substitute x -> r.
    """

    def __init__(self, nfree_max=4, use_scaled_radius=True):
        self._nfree_max = nfree_max
        self._use_scaled_radius = use_scaled_radius

        x, y, r, Rs = symbols('x y r Rs', real=True, positive=True)

        self._locals = {
            "inv": Lambda(x, 1 / x),
            "square": Lambda(x, x * x),
            "cube": Lambda(x, x * x * x),
            "sqrt_abs": Lambda(x, sqrt(Abs(x, evaluate=False))),
            "sqrt": Lambda(x, sqrt(Abs(x, evaluate=False))),
            "log": Lambda(x, log(Abs(x, evaluate=False))),
            "pow": Lambda((x, y), Pow(Abs(x, evaluate=False), y)),
            "exp": Lambda(x, sympify("exp")(x)),
            "abs": Lambda(x, Abs(x, evaluate=False)),
            "x": x,
            "r": r,
            "Rs": Rs,
        }

        self._free_params = [Symbol(f"a{i}", real=True)
                             for i in range(self._nfree_max)]

        for i in range(self._nfree_max):
            self._locals[f"a{i}"] = self._free_params[i]

        self._x = x
        self._r = r
        self._Rs = Rs

    def _substitute_x(self, expr):
        """Substitute x with r/Rs or r depending on use_scaled_radius."""
        if self._use_scaled_radius:
            return expr.subs(self._x, self._r / self._Rs)
        return expr.subs(self._x, self._r)

    @property
    def nfree_max(self):
        """Maximum number of free parameters."""
        return self._nfree_max

    def count_free(self, expr):
        """
        Count the number of free parameters in an expression.

        Parameters
        ----------
        expr : str or SymPy expression

        Returns
        -------
        int
        """
        if isinstance(expr, str):
            if f"a{self._nfree_max}" in expr:
                raise ValueError("Detected more free parameters than allowed.")
            return sum(f"a{i}" in expr for i in range(self._nfree_max))

        free_symbols = expr.free_symbols
        return sum(ai in free_symbols for ai in self._free_params)

    def parse(self, expr_str):
        """
        Parse a string expression into a SymPy expression.

        Parameters
        ----------
        expr_str : str
            Expression like "(a0 * (1 + x)**a1)**-1"

        Returns
        -------
        SymPy expression
        """
        return sympify(expr_str, locals=self._locals)

    def to_c(self, expr_str, function_name="rho"):
        """
        Convert expression to C function code.

        Parameters
        ----------
        expr_str : str
            Expression like "(a0 * (1 + x)**a1)**-1"
        function_name : str, optional
            Name of the C function. Default is "rho".

        Returns
        -------
        str
            C function code.
        """
        expr = self.parse(expr_str)
        nfree = self.count_free(expr)

        # Build parameter list: r, Rs always present (C typedef requires it)
        params = ["r", "Rs"] + [f"a{i}" for i in range(nfree)]
        param_str = ", ".join(f"double {p}" for p in params)

        # Replace x with r/Rs or r
        expr_substituted = self._substitute_x(expr)

        # Generate C code for the expression
        c_expr = ccode(expr_substituted)

        code = f"""double {function_name}({param_str}) {{
    return {c_expr};
}}"""
        return code

    def to_c_header(self, expr_str, function_name="rho"):
        """
        Generate C function declaration (header).

        Parameters
        ----------
        expr_str : str
            Expression string.
        function_name : str, optional
            Name of the C function.

        Returns
        -------
        str
            C function declaration.
        """
        expr = self.parse(expr_str)
        nfree = self.count_free(expr)

        params = ["r", "Rs"] + [f"a{i}" for i in range(nfree)]
        param_str = ", ".join(f"double {p}" for p in params)

        return f"double {function_name}({param_str});"

    def to_c_file(self, expr_str, function_name="rho", include_math=True):
        """
        Generate a complete C source file.

        Parameters
        ----------
        expr_str : str
            Expression string.
        function_name : str, optional
            Name of the C function.
        include_math : bool, optional
            Whether to include math.h. Default is True.

        Returns
        -------
        str
            Complete C source file.
        """
        header = ""
        if include_math:
            header = "#include <math.h>\n\n"

        return header + self.to_c(expr_str, function_name)

    def to_numpy(self, expr_str):
        """
        Convert expression to NumPy function.

        Parameters
        ----------
        expr_str : str
            Expression string.

        Returns
        -------
        callable
            Function with signature f(r, Rs, a0, a1, ...)
        """
        expr = self.parse(expr_str)
        nfree = self.count_free(expr)

        # Replace x with r/Rs or r
        expr_substituted = self._substitute_x(expr)

        if self._use_scaled_radius:
            params = [self._r, self._Rs] + self._free_params[:nfree]
        else:
            params = [self._r] + self._free_params[:nfree]
        return lambdify(params, expr_substituted, "numpy")

    def to_cfunc(self, expr_str, function_name="rho"):
        """
        Compile expression to C on the fly and return a callable.

        The first argument `r` is vectorized (can be array or scalar).
        All other arguments (Rs, a0, a1, ...) must be scalars.

        Parameters
        ----------
        expr_str : str
            Expression string.
        function_name : str, optional
            Name of the C function.

        Returns
        -------
        callable
            Compiled C function: f(r, Rs, a0, a1, ...)
            where r can be array or scalar, others are scalars.
        """
        expr = self.parse(expr_str)
        nfree = self.count_free(expr)

        # Parameters: r is vectorized, rest are scalars
        if self._use_scaled_radius:
            scalar_params = ["Rs"] + [f"a{i}" for i in range(nfree)]
        else:
            scalar_params = [f"a{i}" for i in range(nfree)]
        scalar_param_str = ", ".join(f"double {p}" for p in scalar_params)

        # Replace x with r/Rs or r
        expr_substituted = self._substitute_x(expr)
        c_expr = ccode(expr_substituted)

        # C source code
        sig = f"void {function_name}(double* r, double* out, int n, " \
              f"{scalar_param_str})"
        c_source = f"""
        {sig} {{
            for (int i = 0; i < n; i++) {{
                out[i] = {c_expr.replace('r', 'r[i]')};
            }}
        }}
        """

        # Compile with cffi
        ffi = FFI()
        ffi.cdef(f"{sig};")
        verify_kwargs = dict(libraries=["m"])
        if CFFI_TMPDIR is not None:
            verify_kwargs["tmpdir"] = str(CFFI_TMPDIR)
        lib = ffi.verify(c_source, **verify_kwargs)

        c_func = getattr(lib, function_name)

        def wrapper(r, *args):
            r_arr = np.atleast_1d(np.ascontiguousarray(r, dtype=np.float64))
            result = np.empty(len(r_arr), dtype=np.float64)

            r_ptr = ffi.cast("double*", r_arr.ctypes.data)
            out_ptr = ffi.cast("double*", result.ctypes.data)
            c_func(r_ptr, out_ptr, len(r_arr), *args)

            # Return scalar if input was scalar
            if np.ndim(r) == 0:
                return result[0]
            return result

        return wrapper

    def to_c_array(self, expressions, base_name="rho"):
        """
        Generate C code for multiple expressions as an array of function
        pointers.

        Parameters
        ----------
        expressions : list of str
            List of expression strings.
        base_name : str, optional
            Base name for functions. Functions will be named
            base_name_0, base_name_1, etc.

        Returns
        -------
        str
            C source code with all functions and a function pointer array.
        """
        lines = ["#include <math.h>\n"]

        # Generate all functions
        for i, expr_str in enumerate(expressions):
            func_name = f"{base_name}_{i}"
            lines.append(self.to_c(expr_str, func_name))
            lines.append("")

        # Find max number of parameters across all functions
        max_nfree = max(self.count_free(expr_str) for expr_str in expressions)
        max_params = 2 + max_nfree  # r, Rs, a0, ...

        # Generate typedef for function pointer
        param_types = ", ".join(["double"] * max_params)
        lines.append(f"typedef double (*{base_name}_func)({param_types});")
        lines.append("")

        # Generate array of function pointers
        func_names = [f"{base_name}_{i}" for i in range(len(expressions))]
        lines.append(f"const {base_name}_func {base_name}_functions[] = {{")
        lines.append("    " + ", ".join(func_names))
        lines.append("};")
        lines.append("")
        n_funcs = len(expressions)
        lines.append(f"const int n_{base_name}_functions = {n_funcs};")

        return "\n".join(lines)
