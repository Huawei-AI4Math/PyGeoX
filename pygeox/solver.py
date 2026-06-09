import math

from random import randint
from typing import Optional, Union
from time import time
from copy import deepcopy

import numba
import numpy as np
import sympy
from scipy import optimize
from scipy.spatial.distance import pdist, squareform
from sympy.printing.pycode import PythonCodePrinter

Coordinate = Union[int, float, sympy.Expr]


class SolverNamespace:
    def __init__(self, scene):
        self.scene = scene
        self.status = "not solved"
        self.last_alpha = None
        self.last_beta = None
        self.last_strict_penalty = None
        self.last_strict_tol = None
        self.objective_func = None
        self.vars_to_solve = None
        self.sub_objectives = None
        self.multiple_solutions = None
        self.independent_variables = None
        self._internal_sol = None
        
    """
    Namespace responsible for solving geometric constraint systems
    both analytically and numerically within a geometric scene.

    Provides methods to:
    - Substitute solved values into scene objects.
    - Simplify and prepare equations and constraints.
    - Perform analytical symbolic solving using SymPy.
    - Perform numerical optimization-based solving using SciPy.
    - Manage multiple solutions and pick one.
    - Define and evaluate the objective function for constraints.
    - Analyze and report solution quality and constraint satisfaction.

    Attributes:
        scene: Reference to the geometric scene containing constraints and objects.
        status: Current solver status ("not solved", "success", "failure", etc.).
        multiple_solutions: List of multiple symbolic solutions found (if any).
        _internal_sol: Last internal numerical solution vector.
        other solver-related internal attributes...
    """

    @staticmethod
    def substitute_in_obj(obj, sol, numeric: bool = False, visited=None):
        if visited is None:
            visited = set()
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        # Prepare substitution dict restricted to symbols actually in the expression
        sol_symbols = set(sol.keys())

        for attr in dir(obj):
            if attr in ("__class__", "__dict__", "__weakref__"):
                continue
            attr_descriptor = getattr(type(obj), attr, None)
            if isinstance(attr_descriptor, property) or callable(getattr(obj, attr, None)):
                continue
            try:
                value = getattr(obj, attr)
            except Exception:
                continue
            if isinstance(value, sympy.Basic):
                # Substitute only symbols present in sol and present in value
                relevant_subs = {k: sol[k] for k in value.free_symbols & sol_symbols}
                val = value.subs(relevant_subs)

                if numeric:
                    # Evaluate to float only if fully numeric (no free symbols)
                    if not val.free_symbols:
                        try:
                            val = float(val.evalf())
                        except Exception:
                            pass  # keep symbolic if conversion fails

                try:
                    setattr(obj, attr, val)
                except Exception:
                    pass
            elif hasattr(value, "__dict__"):
                SolverNamespace.substitute_in_obj(value, sol, numeric, visited)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if hasattr(item, "__dict__"):
                        SolverNamespace.substitute_in_obj(item, sol, numeric, visited)

    def _build_mapping_subs(self):
        mapping_subs = {}
        full_parameters = [
            info["symbol"] for _, info in self.scene.parameters.items() if info["full"]
        ]
        for mapping in self.scene.mapping_constraints:
            eq = mapping["equation"]
            if isinstance(eq, sympy.Equality):
                lhs, rhs = eq.lhs, eq.rhs
                if lhs in full_parameters:
                    mapping_subs[lhs] = rhs
        return mapping_subs

    def _substitute_equations(self, mapping_subs):
        equations = []
        for c in self.scene.constraints:
            eq = c["equation"]
            eq_sub = eq.subs(mapping_subs)
            equations.append(eq_sub)
        return equations

    def _extract_bounds_from_symbols(self, symbols):
        """
        Extract bounds from sympy symbols using the domain variable of geoscene.
        Args:
            symbols (list): List of sympy.Symbol objects.
        Returns:
            list: List of (lower, upper) tuples for each symbol.
        """
        bounds = []
        for sym in symbols:
            bound = self.scene.parameters[sym.name]["bound"]
            bounds.append(bound if bound else [-1000,1000] )
            
        return bounds 

    def _random_initial_guess(self, bounds, seed=42):
        """
        Generate a random initial guess within the given bounds.
        This version is corrected to handle None in bounds.
        """
        rng = np.random.default_rng(seed)
        x0 = []
        for low, high in bounds:
            # Check if bounds are valid numbers before using them.
            # This prevents the TypeError with np.isfinite(None).
            is_finite_bound = (
                np.isfinite(low) and np.isfinite(high) if low and high else False
            )
            if is_finite_bound:
                # Ensure low is actually less than high for rng.uniform
                if low < high:
                    x0.append(rng.uniform(low, high))
                else:
                    # If bounds are equal or invalid, just use the value
                    x0.append(low)
            else:
                # Fallback for unbounded variables (e.g., where low or high is None or inf)
                x0.append(rng.uniform(-10, 10))

        return np.array(x0)

    def _check_inequalities(self, sol, mapping_subs, ineqs):
        full_sol = {**sol, **mapping_subs}
        ok = True
        for ineq in ineqs:
            cond = ineq.subs(full_sol)
            if cond is sympy.S.true:
                continue
            if cond is sympy.S.false:
                ok = False
            else:
                ok = False
            if not ok:
                break
        return ok

    def _replace_objects_with_sol_ana(self, sol):
        full_subs = {lhs: rhs.subs(sol) for lhs, rhs in self._mapping_subs.items()}
        sol.update(full_subs)

        self.scene.objects_solved = deepcopy(self.scene.objects)
        for obj_dict in self.scene.objects_solved.values():
            for obj in obj_dict.values():
                SolverNamespace.substitute_in_obj(obj, sol, numeric=True)

        self.scene.solution = sol
        self.status = "success"
        print("[solver.analytical] ✅ Only one solution found. Status: success.")

        return sol

    def analytical(self):
        """
        Attempt to solve the constraint system symbolically (analytically).

        Steps:
            - Substitute mapping constraints.
            - Separate equations and inequalities.
            - Solve equalities using SymPy.
            - Filter and validate solutions.
            - Return unique solution or multiple candidates.

        Raises:
            ValueError: If no solution or invalid solutions found.

        Returns:
            dict or list of dicts: Solution mapping symbols to values.
        """
        self._timings["solver.analytical.start"] = time()

        # 1) build your substitutions and raw equation list
        self._mapping_subs = self._build_mapping_subs()
        equations = self._substitute_equations(self._mapping_subs)

        # 2) split equations vs inequalities
        self._eqs, self._ineqs = [], []
        for expr in equations:
            if isinstance(expr, sympy.Equality):
                self._eqs.append(expr)
            else:
                self._ineqs.append(expr)

        # 3) warn if there are any inequalities
        if self._ineqs:
            print(
                "[WARNING] Inequalities will be ignored for now. They will only be checked when final result is obtained."
            )

        # 4) pick your variables
        reduced = [
            info["symbol"]
            for _, info in self.scene.parameters.items()
            if not info["full"]
        ]
        full_p = [
            info["symbol"] for _, info in self.scene.parameters.items() if info["full"]
        ]
        variables = reduced + [p for p in full_p if p not in self._mapping_subs]

        # 5) solve just the equalities
        raw_solutions = sympy.solve(
            tuple(self._eqs), tuple(variables), dict=True
        )  # sympy.nonlinsolve(eqs, variables)
        # raw_solutions = [dict(zip(variables, sol)) for sol in raw_solutions]
        if not raw_solutions:
            self.status = "failure"
            raise ValueError(
                "[solver.analytical] ❌ No solution found for equalities. Status: failure."
            )

        # filter out solutions where "r" variables are float,int and negative
        filtered_solutions = []
        acceptable_numeric_types = (
            sympy.core.numbers.Float,
            float,
            int,
            sympy.core.numbers.Integer,
        )
        for raw_solution in raw_solutions:
            for var, value in raw_solution.items():
                var_param_type = self.scene.parameters[var.name]["type"]
                if var_param_type == "r" and isinstance(
                    value, acceptable_numeric_types
                ):
                    if value > 0:
                        filtered_solutions.append(raw_solution)

        # 7) branch on number of valid solutions
        if len(filtered_solutions) == 1:
            if not self._check_inequalities(
                self, filtered_solutions[0], self._mapping_subs, self._ineqs
            ):
                self.status = "failure"
                raise ValueError(
                    "[solver.analytical] ❌ Solution failed inequality check. Status: failure."
                )
            self._replace_objects_with_sol_ana(filtered_solutions[0])
            return  self.scene.get_all_objects()

        # multiple solutions
        self.multiple_solutions = filtered_solutions
        # determine independent variables (free symbols in RHS)
        indep = set()
        for sol in filtered_solutions:
            for rhs in sol.values():
                indep |= rhs.free_symbols
        indep_list = sorted(indep, key=lambda s: s.name)
        self._independent_variables = indep_list

        # warn to pick solution
        print(
            f"[solver.analytical] ⚠️ Found {len(filtered_solutions)} solutions. Use pick_solution() to select."
        )
        for idx, sol in enumerate(filtered_solutions):
            print(f"  Solution {idx}: {sol}")
        if indep_list:
            names = ", ".join(str(s) for s in indep_list)
            print(
                f"Independent variables: {names}. You need to fix these to get a unique solution."
            )

        self.status = "multiple_solutions"

        self._timings["solver.analytical.end"] = time()

        return filtered_solutions

    def pick_solution(self, index: int = None, ind_var_values: dict = None):
        """
        Select and apply one solution among multiple symbolic solutions.

        Args:
            index: Index of the solution to pick (random if None).
            ind_var_values: Optional fixed values for independent variables.

        Raises:
            ValueError: If solution contains invalid values or fails inequalities.
        """
        self._timings["solver.pick_solution.start"] = time()

        sols = getattr(self, "multiple_solutions", None)
        if not sols:
            print("[pick_solution] No multiple solutions stored. Nothing to pick.")
            return
        indep = getattr(self, "_independent_variables", None)

        # if index is None pick random
        if not index:
            index = randint(0, len(sols) - 1)
            print(f"[solver.pick_solution] Picked random solution: {index}")

        # choose base solution
        sol_base = sols[index]

        # prepare mapping for independent variables -> if user does not provide values, then pick random values in domain
        mapping = {}
        if indep:
            # use provided values or default to 0
            ind_var_values = ind_var_values or {}
            for var in indep:
                value = ind_var_values.get(var.name, None)
                if not value:
                    indep_param_type = self.scene.parameters[var.name]["type"]
                    if indep_param_type in ["x", "y"]:
                        min, max = self.scene._domain_consts[indep_param_type] or [
                            -10,
                            10,
                        ]
                    else:
                        min = 0
                        max = self.scene._domain_consts["r"][1]
                        if not np.isfinite(max):
                            max = 5
                    value = np.random.uniform(low=min, high=max)
                    print(
                        f"[solver.pick_solution] Picked random value for {var.name}: {value}"
                    )
                mapping[var] = ind_var_values.get(var.name, value)

        # substitute independent var values into solution
        sol_sub = {k: v.subs(mapping) for k, v in sol_base.items()}

        # apply mapping_subs and substitute into scene
        full_sol = {**sol_sub, **self._mapping_subs, **mapping}

        acceptable_numeric_types = (
            sympy.core.numbers.Float,
            float,
            int,
            sympy.core.numbers.Integer,
        )
        if not all(
            isinstance(value, acceptable_numeric_types) for value in full_sol.values()
        ):
            self.status = "failure"
            print(f"[solver.pick_solution] Picked solution is: {full_sol}")
            raise ValueError(
                "[solver.pick_solution] ❌ Solution contains imaginary numbers. Please pick another solution."
            )

        # check if it meets inequality constraint
        if not self._check_inequalities(full_sol, self._mapping_subs, self._ineqs):
            self.status = "failure"
            raise ValueError(
                "[solver.pick_solution] ❌ Solution failed inequality check. Please pick another solution. Status: failure."
            )

        # substitute
        for obj in self.scene.get_all_objects().values():
            SolverNamespace.substitute_in_obj(obj, full_sol, numeric=False)

        self.scene.solution = full_sol
        self.status = "success"

        self._replace_objects_with_sol_ana(sol_sub)

        self._timings["solver.pick_solution.end"] = time()

        # print summary
        if indep:
            names = ", ".join(f"{var.name}={mapping[var]}" for var in indep)
            print(
                f"[pick_solution] ✅ Applied solution {index} with independent variables: {names}"
            )
        else:
            print(f"[pick_solution] ✅ Applied solution {index}.")

        return self.scene.get_all_objects()

    def get_objective_function(
        self,
        variables,
        equations,
        alpha: float = 1.0,
        beta: float = 1.0,
        strict_penalty: float = 1.0,
        strict_tol: float = 1e-4,
        use_numba: bool = False,
        objective_type: str = "SSE",
        T: float = 1.0,
    ):
        """
        Build the objective function for numerical optimization from constraints.

        Supports both pure Python and Numba-jitted versions.

        Args:
            variables: List of variables to optimize.
            equations: List of (description, sympy expression) constraints.
            alpha: Weight for equality constraints.
            beta: Weight for inequality constraints.
            strict_penalty: Penalty for strict inequality violations.
            strict_tol: Tolerance for strict inequalities.
            use_numba: If True, return a Numba-accelerated function.

        Returns:
            tuple: (objective function, variables, dictionary of sub-objectives)
        """
        # Store parameters
        self.alpha = alpha
        self.beta = beta
        self.strict_penalty = strict_penalty
        self.strict_tol = strict_tol

        # Classify constraints
        eqs, geqs, stricts, neqs = [], [], [], []
        for desc, expr in equations:
            if isinstance(expr, sympy.Equality):
                eqs.append((desc, expr.lhs - expr.rhs))
            elif isinstance(expr, sympy.GreaterThan):
                geqs.append((desc, expr.lhs - expr.rhs))
            elif isinstance(expr, sympy.LessThan):
                geqs.append((desc, expr.rhs - expr.lhs))
            elif isinstance(expr, sympy.StrictGreaterThan):
                stricts.append((desc, expr.lhs - expr.rhs))
            elif isinstance(expr, sympy.StrictLessThan):
                stricts.append((desc, expr.rhs - expr.lhs))
            elif isinstance(expr, sympy.Unequality):
                neqs.append((desc, expr.lhs - expr.rhs))
            else:
                eqs.append((desc, expr))

        # If numba requested, dynamically generate and jitt the function
        if use_numba:
            # Remap symbols to x0, x1, ...
            n = len(variables)
            x_syms = sympy.symbols(f"x0:{n}")
            subs_map = {variables[i]: x_syms[i] for i in range(n)}
            eq_subs = [e.subs(subs_map) for _, e in eqs]
            geq_subs = [e.subs(subs_map) for _, e in geqs]
            strict_subs = [e.subs(subs_map) for _, e in stricts]
            neq_subs = [e.subs(subs_map) for _, e in neqs]

            printer = PythonCodePrinter()
            lines = []

            # unpack x
            for i in range(n):
                lines.append(f"    x{i} = x[{i}]")

            # compute constraints
            for i, e in enumerate(eq_subs):
                lines.append(f"    eq_{i} = {printer.doprint(e)}")
            for i, e in enumerate(geq_subs):
                lines.append(f"    geq_{i} = {printer.doprint(e)}")
            for i, e in enumerate(strict_subs):
                lines.append(f"    str_{i} = {printer.doprint(e)}")
            for i, e in enumerate(neq_subs):
                lines.append(f"    neq_{i} = {printer.doprint(e)}")

            # equality penalty (unchanged)
            eq_terms = (
                " + ".join(f"eq_{i}*eq_{i}" for i in range(len(eq_subs))) or "0.0"
            )
            lines.append(f"    eq_penalty = {eq_terms}")

            # inequality (geq) penalty without max()
            lines.append("    geq_penalty = 0.0")
            for i in range(len(geq_subs)):
                lines.append(f"    if -geq_{i} > 0.0:")
                lines.append(f"        geq_penalty += (-geq_{i})**2")

            # strict-inequality penalty without max()
            lines.append("    strict_penalty_val = 0.0")
            for i in range(len(strict_subs)):
                lines.append(f"    if {strict_tol} - str_{i} > 0.0:")
                lines.append(
                    f"        strict_penalty_val += ({strict_tol} - str_{i})**2"
                )

            # not-equal penalty loop (unchanged)
            lines.append("    neq_penalty = 0.0")
            for i in range(len(neq_subs)):
                lines.append(f"    if abs(neq_{i}) < {strict_tol}:")
                lines.append(f"        neq_penalty += {strict_penalty}")

            # assemble return based on objective_type
            ret = f"{alpha}*eq_penalty + {beta}*geq_penalty + {beta}*strict_penalty_val + neq_penalty"
            if objective_type == "SSE":
                lines.append(f"    return {ret}")
            elif objective_type == "Log":
                lines.append(f"    return np.log(1.0 + {ret})")
            elif objective_type == "Log-Sum-Exp":
                lines.append(f"    T_inv = 1.0 / {T}")
                lines.append("    total_satisfaction = 0.0")
                for i in range(len(eq_subs)):
                    lines.append(f"    total_satisfaction += np.exp(- (eq_{i}*eq_{i}) * T_inv)")
                for i in range(len(geq_subs)):
                    lines.append(f"    if -geq_{i} > 0.0:")
                    lines.append(f"        total_satisfaction += np.exp(- ((-geq_{i})*(-geq_{i})) * T_inv)")
                    lines.append(f"    else:")
                    lines.append(f"        total_satisfaction += 1.0")
                for i in range(len(strict_subs)):
                    lines.append(f"    if {strict_tol} - str_{i} > 0.0:")
                    lines.append(f"        total_satisfaction += np.exp(- (({strict_tol} - str_{i})*({strict_tol} - str_{i})) * T_inv)")
                    lines.append(f"    else:")
                    lines.append(f"        total_satisfaction += 1.0")
                for i in range(len(neq_subs)):
                    lines.append(f"    if abs(neq_{i}) < {strict_tol}:")
                    lines.append(f"        total_satisfaction += np.exp(- ({strict_penalty}*{strict_penalty}) * T_inv)")
                    lines.append(f"    else:")
                    lines.append(f"        total_satisfaction += 1.0")
                n_total = len(eq_subs) + len(geq_subs) + len(strict_subs) + len(neq_subs)
                lines.append(f"    return np.log({float(n_total)}) - np.log(total_satisfaction + 1e-12)")
            else:
                raise ValueError(f"Unknown objective_type: {objective_type}")

            func_src = (
                "@numba.njit(fastmath=True, parallel=True, error_model='numpy')\n"
                "def objective(x):\n" + "\n".join(lines)
            )
            func_src = func_src.replace("math.", "np.")

            # assemble and exec
            namespace = {"numba": numba, "abs": abs, "math": math, "np": np}
            local_ns = {}
            exec(func_src, namespace, local_ns)
            objective_fn = local_ns["objective"]

            # build lambdas for diagnostics
            eq_lmb = [(d, sympy.lambdify(variables, e, "numpy"), e) for d, e in eqs]
            geq_lmb = [(d, sympy.lambdify(variables, e, "numpy"), e) for d, e in geqs]
            strict_lmb = [
                (d, sympy.lambdify(variables, e, "numpy"), e) for d, e in stricts
            ]
            neq_lmb = [(d, sympy.lambdify(variables, e, "numpy"), e) for d, e in neqs]
            sub_objs = {
                "equality": eq_lmb,
                "geq": geq_lmb,
                "strict": strict_lmb,
                "neq": neq_lmb,
            }
            return objective_fn, variables, sub_objs

        # Fallback: pure-Python lambdified version
        eq_lambdas = [(d, sympy.lambdify(variables, e, "numpy"), e) for d, e in eqs]
        geq_lambdas = [(d, sympy.lambdify(variables, e, "numpy"), e) for d, e in geqs]
        strict_lambdas = [
            (d, sympy.lambdify(variables, e, "numpy"), e) for d, e in stricts
        ]
        neq_lambdas = [(d, sympy.lambdify(variables, e, "numpy"), e) for d, e in neqs]

        def objective(x):
            eq_vals = [f(*x) for _, f, _ in eq_lambdas]
            geq_vals = [f(*x) for _, f, _ in geq_lambdas]
            strict_vals = [f(*x) for _, f, _ in strict_lambdas]
            neq_vals = [f(*x) for _, f, _ in neq_lambdas]
            eq_penalty = sum(np.square(v) for v in eq_vals)
            geq_penalty = sum(np.square(max(0, -v)) for v in geq_vals)
            str_pen = sum(np.square(max(0, strict_tol - v)) for v in strict_vals)
            neq_penalty = sum(
                strict_penalty if abs(v) < strict_tol else 0 for v in neq_vals
            )
            sse = alpha * eq_penalty + beta * geq_penalty + beta * str_pen + neq_penalty
            
            if objective_type == "SSE":
                return sse
            elif objective_type == "Log":
                return np.log(1.0 + sse)
            elif objective_type == "Log-Sum-Exp":
                T_inv = 1.0 / T
                total_satisfaction = 0.0
                for v in eq_vals:
                    total_satisfaction += np.exp(-(v * v) * T_inv)
                for v in geq_vals:
                    total_satisfaction += np.exp(-(max(0, -v)**2 * T_inv)) if v < 0 else 1.0
                for v in strict_vals:
                    violation = max(0, strict_tol - v)
                    total_satisfaction += np.exp(-(violation * violation * T_inv)) if violation > 0 else 1.0
                for v in neq_vals:
                    total_satisfaction += np.exp(-(strict_penalty * strict_penalty * T_inv)) if abs(v) < strict_tol else 1.0
                n_total = len(eq_vals) + len(geq_vals) + len(strict_vals) + len(neq_vals)
                return np.log(float(n_total)) - np.log(total_satisfaction + 1e-12)
            else:
                raise ValueError(f"Unknown objective_type: {objective_type}")

        sub_objectives = {
            "equality": eq_lambdas,
            "geq": geq_lambdas,
            "strict": strict_lambdas,
            "neq": neq_lambdas,
        }
        return objective, variables, sub_objectives

    def _replace_objects_with_sol_num(self, vars_to_solve, internal_sol, mapping_subs, red_full_mapping):
        self._internal_sol = internal_sol  # solution

        if vars_to_solve is None: #all values given numerically?
            sol = mapping_subs
            #replace red_full_mapping
            full_subs = {
                lhs: rhs.subs(sol) for lhs, rhs in red_full_mapping.items()
            }  # replace all expressions including the replaced original variables
            sol.update(full_subs)

        else:
            sol = dict(zip(vars_to_solve, internal_sol)) # organize for return and geoscene

            #replace mapping_subs (circumradius = 5 )
            full_subs = {
                lhs: rhs.subs(sol) for lhs, rhs in mapping_subs.items()
            }
            sol.update(full_subs)

            #replace red_full_mapping
            full_subs = {
                lhs: rhs.subs(sol) for lhs, rhs in red_full_mapping.items()
            }  # replace all expressions including the replaced original variables
            sol.update(full_subs)
            
        # Save the original scene (not solved)
        self.scene.objects_solved = {k: {kk: deepcopy(vv) for kk, vv in v.items()} 
                                    for k, v in self.scene.objects.items()}
        
        # Apply substitution directly
        for obj_dict in self.scene.objects_solved.values():
            for obj in obj_dict.values():
                SolverNamespace.substitute_in_obj(obj, sol, numeric=True)

        # Apply substitution to tracked quantities
        self.scene.track_solved = {}  # Initialize/clear the solved tracking dict
        for track_name, track_info in self.scene.track_unsolved.items():
            expr = track_info["quantity"]
            
            # Substitute the numerical solution into the expression
            substituted_expr = expr.subs(sol)
            
            # Evaluate the result to a Python float
            try:
                numerical_value = float(substituted_expr.evalf())
            except TypeError:
                # This can happen if the expression isn't fully numeric
                # (e.g., sol was incomplete for this expression)
                print(f"[Warning] Tracked quantity '{track_name}' could not be fully evaluated to a number.")
                numerical_value = substituted_expr # Store the partially-substituted expression
            
            self.scene.track_solved[track_name] = numerical_value

        self.scene.solution = sol

        return sol

    #TODO
    #add penalty for line segments to be separated by min_dist
    def _add_penalty(self, distance_penalty, min_dist):
        # Early return if distance_penalty is 0 or negative
        if distance_penalty <= 0:
            return

        #1. Make area of all closed shapes larger then min_dist**2
        #scanned_points = []
        for obj_type, obj_list in self.scene.objects.items():                
            if obj_type not in ["Point", "LineSegment", "Line", "Ray", "MajorArc", "Angle"]:
                for obj in obj_list.values():
                    self.scene.constraint.gt(distance_penalty * ( obj.area - min_dist**2 ), 0 , f"Area of {obj.name} > {min_dist}**2 (penalty)")
                    #if obj_type == "Circle":
                    #    points_to_add = [obj.center]
                    #elif obj_type in ["MinorArc"]:
                    #    points_to_add = [obj.center, obj.start_point, obj.end_point]
                    #else:
                    #    points_to_add = obj.points

        ##2. deal with infinite lines -> force them to have seperate slope + intercept pairs
        #line_objs = list(self.scene.objects["Line"].values())

        #for i in range(len(line_objs)):
        #    li = line_objs[i]
        #    for j in range(i + 1, len(line_objs)):
        #        lj = line_objs[j]

        #        ds = li.slope - lj.slope
        #        db = li.intercept - lj.intercept

        #        self.scene.constraint.gt(
        #            ds + db  - (2 * min_dist),
        #            0,
        #            f"Separate Line {li.name} and {lj.name}: (penalty)"
        #        )

        ##3. deal with moving point animation...
        points_to_remove = self.scene._moving_objects

        ##4. penalize distance between free_points and all_points(except moving points)
        all_points = list(self.scene.objects["Point"].values())
        all_points = [p for p in all_points if p not in points_to_remove]
        #free_points = [p for p in all_points if p not in scanned_points]

        seen = set()
        for p1 in all_points:
            for p2 in all_points:
                if p1 == p2:
                    continue
                pair = frozenset((p1, p2))
                if pair in seen:
                    continue
                seen.add(pair)

                # Only add constraint if distance_penalty > 0
                if distance_penalty > 0:
                    self.scene.constraint.gt(
                        distance_penalty * (p1.distance_squared(p2) - min_dist**2),
                        0,
                        f"{p1.name} {p2.name} > {min_dist} (penalty)")


        #from itertools import combinations

        #for point1, point2 in combinations(
        #    list(self.scene.objects["Point"].values()), 2
        #):
            # avoid overlapping points
        #    eq = distance_penalty * (point1.distance_squared(point2) - min_dist**2)
        #    if not isinstance(eq, float):
        #        self.scene.constraint.gt(
        #            distance_penalty
        #            * (point1.distance_squared(point2) - min_dist**2),
        #            0,
        #            f"{point1.name} {point2.name} > {min_dist}  (penalty)",
        #        )
        # prevent "radius" variables from being 0
        #r_symbols = [
        #    param["symbol"]
        #    for param in self.scene.parameters.values()
        #    if param["type"] == "r"
        #]
        #[
        #    self.scene.constraint.gt(r_symbol, min_dist)
        #    for r_symbol in r_symbols
        #    if not isinstance(r_symbol, float)
        #]


    def simplify_constraints(self, max_linear = 0):
        """
        Simplify the constraint system by applying known variable mappings, removing
        trivial equalities, and optionally performing cheap linear eliminations.

        This reduces the number of independent variables and constraints, aiding solver efficiency.

        Args:
            max_linear (int): Maximum number of additional cheap linear eliminations to attempt.

        Returns:
            tuple:
                - indep_vars (list[sympy.Symbol]): Independent variables after simplification.
                - reduced_eqs (list[sympy.Expr]): Simplified constraint expressions (lhs - rhs).
                - mapping (dict[sympy.Symbol, sympy.Expr]): Mapping from dependent to independent variables.
                - red_full_mapping (dict[sympy.Symbol, sympy.Expr]): Final mapping of variables to expressions.
        """

        params = [p["symbol"] for p in self.scene.parameters.values()]
        red_full_mapping = {p["equation"].lhs: p["equation"].rhs for p in self.scene.mapping_constraints}

        #seperate equalities and inequalities and remove duplicates
        eqs = list(set([(c["description"], c["equation"]) for c in self.scene.constraints if isinstance(c["equation"], sympy.Equality)]))
        ineqs = list(set([(c["description"], c["equation"]) for c in self.scene.constraints if not isinstance(c["equation"], sympy.Equality)]))

        #Step 1. easy simplify (remove square roots)
        def has_sqrtish(expr):
            if expr.has(sympy.sqrt):
                return True
            for p in expr.atoms(sympy.Pow):
                if p.exp == sympy.Rational(1, 2):
                    return True
            return False

        for i, (desc, eq) in enumerate(eqs):
            if has_sqrtish(eq.lhs) or has_sqrtish(eq.rhs):
                eqs[i] = (desc, sympy.Eq(eq.lhs**2, eq.rhs**2))
        
        #Step 2. deal with reduced mapping variables for regular polygons, etc...
        eqs = [(desc,sympy.Eq(eq.lhs.xreplace(red_full_mapping), eq.rhs.xreplace(red_full_mapping))) for desc, eq in eqs ]
        ineqs = [(d, rel.func(rel.lhs.xreplace(red_full_mapping),rel.rhs.xreplace(red_full_mapping))) for d, rel in ineqs]
        params = [p for p in params if p not in red_full_mapping] #remove mapped variables from full_params
        dep_indep_mapping =  {} #red_full_mapping

        # Step 3: prioritize equations where one side is fully numeric first, then by equation size (ops+len)
        def is_atomic_numeric(x):
            return isinstance(x, (int, float, sympy.Integer, sympy.Float)) or (x.is_number and not x.free_symbols)

        def is_numeric_expr(e):
            return is_atomic_numeric(e.lhs) or is_atomic_numeric(e.rhs)

        def eq_size_metric(e):
            # cost: total operations 
            return sympy.count_ops(e.lhs - e.rhs, visual=False)

        eqs.sort(key=lambda eq: (not is_numeric_expr(eq[1]), eq_size_metric(eq[1])))


        #step 4. find dependent variables - simple - remove eqs

        new_eqs = []
        for desc, eq in eqs:
            L, R = eq.lhs, eq.rhs
            if isinstance(L, sympy.Symbol) and ( (not R.has(L)) and ( not ( R.free_symbols & set(dep_indep_mapping.keys())))):
                if L not in dep_indep_mapping:
                    dep_indep_mapping[L] = R
                    #make sure no dep variable exists in R
                    continue
            elif isinstance(R, sympy.Symbol) and not ( (not L.has(R)) and (not ( L.free_symbols & set(dep_indep_mapping.keys())))):
                if R not in dep_indep_mapping:
                    dep_indep_mapping[R] = L
                    continue
            new_eqs.append((desc,eq))
        eqs = new_eqs



        #step 5. replace in all equations and inequalities the dependent vars
        if dep_indep_mapping:
            eqs = [ (desc,  sympy.Eq(eq.lhs.xreplace(dep_indep_mapping), eq.rhs.xreplace(dep_indep_mapping))) for desc,eq in eqs]
            ineqs =[ (desc, rel.func( rel.lhs.xreplace(dep_indep_mapping), rel.rhs.xreplace(dep_indep_mapping))) for desc, rel in ineqs ]



        # 5) optional: do a few *cheap* linear eliminations (at most `max_linear`)
        taken = 0
        remaining = []
        for desc, eq in eqs:
            if taken >= max_linear:
                remaining.append((desc,eq))
                continue
            pick = _linear_in_one_eval(eq)
            if pick is None:
                remaining.append((desc,eq))
                continue
            v, rhs = pick
            dep_indep_mapping[v] = rhs #TODO: make sure rhs does not have any dependent variables???
            taken += 1

        #6) simplify and remove true equations

        #SIMPLIFY_FAST = lambda e: sympy.powsimp(sympy.expand_mul(e), force=False)
        simplify_func = lambda e: sympy.expand_mul(e)
        #SIMPLIFY_NONE = lambda e: e
        new_eqs = []
        for desc, eq in eqs:
            lhs = eq.lhs
            rhs = eq.rhs
            if max_linear:
                lhs = lhs.xreplace(dep_indep_mapping)
                rhs = rhs.xreplace(dep_indep_mapping)
            
            simplified = simplify_func(lhs - rhs)
            # Drop trivial 0 == 0
            if simplified == 0:
                continue
            new_eqs.append((desc, simplified))
        eqs = new_eqs

        new_ineqs = []
        for desc, rel in ineqs:
            if rel == sympy.S.true:
                continue

            lhs = rel.lhs
            rhs = rel.rhs
            if max_linear:
                lhs = rel.lhs.xreplace(dep_indep_mapping)
                rhs = rel.rhs.xreplace(dep_indep_mapping)
            new_rel = rel.__class__(simplify_func(lhs), simplify_func(rhs))  # rebuild preserving comparator

            if new_rel == sympy.S.true:
                continue

            new_ineqs.append((desc, new_rel))
        ineqs = new_ineqs

        #merge equalities and inequalities
        eqs = remaining + ineqs
        
        dep_vars = list(dep_indep_mapping.keys())
        indep_vars = [p for p in params if p not in dep_vars]

        return indep_vars, eqs, dep_indep_mapping, red_full_mapping


    #TODO Later -> add support for dynamic regular polygons
    #Now -> Can mode all points that are not in regular polygon 
    def numerical_fast(self, new_x, fixed_vars_idx: list = []):

        
        # Find indices where values differ between self._dynamic_sol and new_x
        diff_indices = []
        fixed_indices = []
        for i in range(len(self._internal_sol)):
            if (abs(self._internal_sol[i] - new_x[i]) > 1e-6) or (i in fixed_vars_idx):  # tolerance for floating point comparison
                fixed_indices.append(i)
            else:
                diff_indices.append(i)
        

         # Pre-allocate x_full (reused in every objective evaluation)
        x_full = self._internal_sol.copy()

        def constrained_objective(x_reduced):
            # update in place (vectorized)
            x_full[diff_indices] = x_reduced
            x_full[fixed_indices] = new_x[fixed_indices]
            return self.objective_func(x_full)

        reduced_bounds = [self._bounds[i] for i in diff_indices]

        # Initial guess is the current values at the differing indices
        self._internal_sol_reduced = self._internal_sol[diff_indices]
        
        # Only optimize if there are variables to optimize
        if len(diff_indices) > 0:
            result = optimize.minimize(
                constrained_objective,
                self._internal_sol_reduced,
                method="L-BFGS-B",
                bounds=reduced_bounds,
                options = {"maxiter": 100, "ftol": 1e-6, "gtol": 1e-6}
            )
            if result.success and result.fun < 1e-4:
                # Update only the optimized values
                for i, idx in enumerate(diff_indices):
                    self._internal_sol[idx] = result.x[i]
                # Update the fixed values to match new_x
                for idx in fixed_indices:
                    self._internal_sol[idx] = new_x[idx]

                self._replace_objects_with_sol_num(self.vars_to_solve, self._internal_sol, {}, self._red_full_mapping) #TODO<-bottleneck..... speed this up later
            else:
                print(f"[WARNING] Optimization failure in numerical_fast: result.success: {result.success};  result.fun: {result.fun}; result.x: {result.x}")


    def numerical(
        self,
        method="basinhoping",
        seed=None,
        alpha=1.0,
        beta=1.0,
        strict_penalty=1.0,
        strict_tol=1e-4,
        n_iter=1000,
        step_size=0.5,
        T=2,
        distance_penalty: float = 1,
        min_dist: Optional[float] = None,
        use_numba: bool = True,
        save_param_history: bool = False,
        verbose: bool = True,
        just_generate_objective_function: bool = False,
        objective_type: str = "SSE",
        objective_T: float = 1.0,
    ):
        """
        Solve the constraint system numerically using global or local optimization.

        Supports methods like basinhopping, L-BFGS-B, and dual_annealing.

        Args:
            method: Optimization method name.
            seed: Random seed.
            alpha, beta, strict_penalty, strict_tol: Penalty weights and tolerances.
            distance_penalty: Weight for penalty avoiding degenerate configurations.
            min_dist: Minimum allowed distance between points.
            use_numba: Use Numba-accelerated objective function.
            save_param_history: Save parameter values during optimization.
            verbose: Print detailed solver progress.
            objective_type: Type of objective function. "SSE" (default), "Log", or "Log-Sum-Exp".
            objective_T: Temperature for Log-Sum-Exp objective (only used when objective_type="Log-Sum-Exp").

        Returns:
            dict: Dictionary with all objects with scene and solved numerical values.

        Raises:
            ValueError: If optimization fails.
        """
        #Todo: add some warning if user called this function directly
        #if self.scene._moving_objects:
        #    raise RuntimeWarning("This scene contains moving points. Please run scene.animate.run() instead.")
        
        #add timing
        self.scene._timings["solve.numerical.init"] = time() 

        if distance_penalty > 0:
            # compute default threshold if not provided
            if min_dist is None:
                if not self.scene._domain_consts["x"]:
                    min_dist = 0.1
                else:
                    lx, ux = self.scene._domain_consts["x"]
                    ly, uy = self.scene._domain_consts["y"]
                    min_dist = ((ux - lx) + (uy - ly)) / 200

            # print(
            #     f"[INFO] Penalty active to prevent overlapping points (min_dist={min_dist}). To remove add: distance_penalty = 0"
            # )

            self._add_penalty(distance_penalty, min_dist)

        #print(f"[INFO] Starting constraint simplification now.")
        #init_vars_n  = len(self.scene.parameters)
        #init_cons_n  = len(self.scene.constraints)
        #variables, equations, mapping_subs, red_full_mapping =  self.simplify_constraints()
        #print(f"[INFO] Simplification complete. Reduced free variables from {init_vars_n} to {len(variables)} and constraints from {init_cons_n} to {len(equations)}.")
        #print(f"[INFO] Optimizing now...")


        # Common setup: build substitution map and collect variables
        self._red_full_mapping = self._build_mapping_subs()
        reduced = [
            info["symbol"]
            for _, info in self.scene.parameters.items()
            if not info["full"]
        ]
        full = [
            info["symbol"] for _, info in self.scene.parameters.items() if info["full"]
        ]
        variables = reduced + [p for p in full if p not in self._red_full_mapping]
        equations = [(eq["description"], eq["equation"].subs(self._red_full_mapping)) for eq in self.scene.constraints]
        mapping_subs = {}
        
        self.objective_func, self.vars_to_solve, self.sub_objectives = (
            self.get_objective_function(
                variables = variables,
                equations = equations,
                alpha=alpha,
                beta=beta,
                strict_penalty=strict_penalty,
                strict_tol=strict_tol,
                use_numba=use_numba,
                objective_type=objective_type,
                T=objective_T,
            )
        )

        if just_generate_objective_function:
            return 

        self._bounds = self._extract_bounds_from_symbols(self.vars_to_solve)

        if not self.vars_to_solve:
            print("[WARNING] No variables to solve")
            sol = self._replace_objects_with_sol_num(None, None, {}, self._red_full_mapping)
            self.numerical_analysis(verbose=verbose)
            return sol

        # initial guess is previously found solution or random value
        x0 = (
            self._internal_sol
            if self._internal_sol is not None
            else self._random_initial_guess(self._bounds, seed)
        )

        self.scene._timings["solve.numerical.optimitize.start"] = time() 

        if method == "basinhoping":
            minimizer_kwargs = {"method": "L-BFGS-B", "bounds": self._bounds}

            # List to store parameter values at each iteration
            if save_param_history:
                self.history_params = []
                self.history_obj_val = []

                def basinhopping_callback(x_current_best: np.ndarray):
                    self.history_params.append(x_current_best.copy())
                    self.history_obj_val.append(self.objective_func(x_current_best))

                minimizer_kwargs["callback"] = basinhopping_callback

            bh_res = optimize.basinhopping(
                self.objective_func,
                x0,
                niter=n_iter,
                stepsize=step_size,
                T=T,
                seed=seed,
                minimizer_kwargs=minimizer_kwargs,
            )
            result = bh_res.lowest_optimization_result

        elif method == "L-BFGS-B":
            result = optimize.minimize(
                self.objective_func,
                x0,
                method = "L-BFGS-B",
                bounds = self._bounds,
                options = {"maxiter": 200, "ftol": 1e-12, "gtol": 1e-08},
            )
        elif method == 'dual_annealing':
            result = optimize.dual_annealing(
                func=self.objective_func,
                x0 = x0,
                bounds= self._bounds,
                seed=seed,
                maxiter=2000 
            )
        else:
            raise ValueError("[solver.numerical] Optimization method not implemented")
        self.scene._timings["solve.numerical.optimitize.end"] = time() 

        if not result.success:
            raise ValueError("[solver.numerical] Optimization failed.")
        
        self.scene._timings["solve.numerical.replacement.start"] = time() 
        sol = self._replace_objects_with_sol_num(variables, result.x, mapping_subs, self._red_full_mapping)
        self.scene._timings["solve.numerical.replacement.end"] = time() 
        
        self.scene._timings["solve.numerical.analysis.start"] = time() 
        self.numerical_analysis(verbose=verbose)
        self.scene._timings["solve.numerical.analysis.end"] = time() 

        return self.scene.get_all_objects()

    def has_any_overlapping_points(self, points_info = None, threshold=0.0001):
        if points_info is None:
            point_coords_dict = self.scene.get_points()
        else:
            point_coords_dict = points_info
        coords = np.array(list(point_coords_dict.values()), dtype=float)
        distances_condensed = pdist(coords)
        dist_matrix = squareform(distances_condensed)
        np.fill_diagonal(dist_matrix, np.inf)
        return np.any(dist_matrix < threshold)

    def numerical_analysis(self, x=None, verbose=True, points_info=None):
        """
        Evaluate and report the quality of a numerical solution.

        Prints objective value, satisfaction of equality, inequality, strict,
        and not-equal constraints, and warns about overlapping points.

        Args:
            x: Optional solution vector to evaluate.
            verbose: Whether to print detailed report.

        Returns:
            dict: Analysis summary including objective and constraint values.
        """
        if x is None:
            x = self._internal_sol

        def eval_subs(subs):
            return [(desc, expr, f(*x)) for (desc, f, expr) in subs]

        eq_vals = eval_subs(self.sub_objectives["equality"])
        geq_vals = eval_subs(self.sub_objectives["geq"])
        strict_vals = eval_subs(self.sub_objectives["strict"])
        neq_vals = eval_subs(self.sub_objectives["neq"])
        obj_val = self.objective_func(x)
        # penalty_val = self.penalty_func(x) if self.penalty_func else None

        # Status logic
        if np.isnan(obj_val) or np.isinf(obj_val):
            self.status = "failure"
        elif obj_val < self.strict_tol:
            self.status = "success"
        else:
            self.status = "failure"

        if verbose:
            total_width = 40 + 30 + 12 + 1 + 3

            def print_section(title, vals, check_fn, symbol):
                if not vals:
                    return  # Do not print empty sections

                print(f"\n{title}")
                print("-" * total_width)
                # Adjusted header formatting
                print(
                    f"{'Name':<40} {'Equation':<30} {'Value':>12}  {' '}"
                )  # Added extra space for checkmark column
                print("-" * total_width)
                for desc, eqn, val in vals:
                    ok = check_fn(val)
                    check = "✅" if ok else "❌"
                    desc_str = str(desc)[:40]
                    eqn_str = str(eqn)[:30]
                    print(f"{desc_str:<40} {eqn_str:<30} {val:12.4e}  {check}")

            print("=" * total_width)
            print(f"Objective function value: {obj_val:.4e} | Status: {self.status}")
            print("=" * total_width)

            print_section(
                "Equality constraints (should be ≈ 0):",
                eq_vals,
                lambda v: abs(v) < self.strict_tol,
                "≈",
            )
            print_section(
                "Inequality constraints (should be ≥ 0):",
                geq_vals,
                lambda v: v > -self.strict_tol,
                "≥",
            )
            print_section(
                "Strict constraints (should be > 0):",
                strict_vals,
                lambda v: v > -self.strict_tol,
                ">",
            )
            print_section(
                "Not-equal constraints (should be ≠ 0):",
                neq_vals,
                lambda v: abs(v) > self.strict_tol,
                "≠",
            )
            print("=" * total_width)

        if self.has_any_overlapping_points(points_info):
            if verbose:
                print(
                    "[WARNING] Overlapping points found. To avoid overlapping points, please consider adding a penalty function to the numerical solver or changing penalty_type."
                )
            has_overlapping_points = True
        else:
            has_overlapping_points = False

        return {
            "objective": obj_val,
            "status": self.status,
            "equalities": [{"desc": desc, "eqn": eqn, "val": val, "pass": abs(val) < self.strict_tol} for desc, eqn, val in eq_vals],
            "geq": [{"desc": desc, "eqn": eqn, "val": val, "pass": val > -self.strict_tol} for desc, eqn, val in geq_vals],
            "strict": [{"desc": desc, "eqn": eqn, "val": val, "pass": val > -self.strict_tol} for desc, eqn, val in strict_vals],
            "neq": [{"desc": desc, "eqn": eqn, "val": val, "pass": abs(val) > self.strict_tol} for desc, eqn, val in neq_vals],
            "has_overlapping_points": has_overlapping_points,
        }
