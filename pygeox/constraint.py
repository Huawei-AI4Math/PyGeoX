from typing import Optional, Union

import sympy
from typeguard import typechecked

# A type alias for things that can be used in a sympy expression
SympyOperand = Union[sympy.Expr, int, float]


@typechecked
class ConstraintNamespace:
    def __init__(self, scene):
        self.scene = scene

    def eq(
        self, a: SympyOperand, b: SympyOperand, description: Optional[str] = None
    ) -> Optional[sympy.Eq]:
        """Adds an equality constraint (a == b) to the scene.
        If 'a' is a square root expression, it squares both sides.
        """
        equation = sympy.Eq(a, b)
        desc = description or f"{a} == {b}"

        #TODO: improve logic here, decide if we should add or not (maybe simplify)?, deal with square root on both sides
        if equation == sympy.S.true:
            # print(f"[WARNING] Condition {desc} already met.")
            return None

        self.scene.add_constraint(equation, desc)
        return equation

    def neq(
        self, a: SympyOperand, b: SympyOperand, description: Optional[str] = None
    ) -> Optional[sympy.Ne]:
        """Adds an inequality constraint (a != b) to the scene."""
        equation = sympy.Ne(a, b)
        desc = description or f"{a} != {b}"

        if isinstance(equation, sympy.logic.boolalg.BooleanTrue):
            # print(f"[WARNING] Condition {desc} already met.")
            return None
        
        self.scene.add_constraint(equation, desc)
        return equation

    def gt(
        self, a: SympyOperand, b: SympyOperand, description: Optional[str] = None
    ) -> Optional[sympy.Gt]:
        """Adds a 'greater than' constraint (a > b) to the scene."""
        equation = sympy.Gt(a, b)
        desc = description or f"{a} > {b}"

        if isinstance(equation, sympy.logic.boolalg.BooleanTrue):
            # print(f"[WARNING] Condition {desc} already met.")
            return None
    
        self.scene.add_constraint(equation, desc)
        return equation

    def lt(
        self, a: SympyOperand, b: SympyOperand, description: Optional[str] = None
    ) -> Optional[sympy.Lt]:
        """Adds a 'less than' constraint (a < b) to the scene."""
        equation = sympy.Lt(a, b)
        desc = description or f"{a} < {b}"

        if isinstance(equation, sympy.logic.boolalg.BooleanTrue):
            # print(f"[WARNING] Condition {desc} already met.")
            return None

        self.scene.add_constraint(equation, desc)
        return equation

    def geq(
        self, a: SympyOperand, b: SympyOperand, description: Optional[str] = None
    ) -> Optional[sympy.Ge]:
        """Adds a 'greater than or equal to' constraint (a >= b) to the scene."""
        equation = sympy.Ge(a, b)
        desc = description or f"{a} >= {b}"
        
        if isinstance(equation, sympy.logic.boolalg.BooleanTrue):
            # print(f"[WARNING] Condition {desc} already met.")
            return None

        self.scene.add_constraint(equation, desc)
        return equation

    def leq(
        self, a: SympyOperand, b: SympyOperand, description: Optional[str] = None
    ) -> Optional[sympy.Le]:
        """Adds a 'less than or equal to' constraint (a <= b) to the scene."""
        equation = sympy.Le(a, b)
        desc = description or f"{a} <= {b}"

        if isinstance(equation, sympy.logic.boolalg.BooleanTrue):
            # print(f"[WARNING] Condition {desc} already met.")
            return None
    
        self.scene.add_constraint(equation, desc)
        return equation
