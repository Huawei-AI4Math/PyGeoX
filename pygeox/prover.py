import os
os.environ['USE_SYMENGINE'] = '1'

from sympy import symbols, Poly, prem, degree, factor, solve
import sympy
import re

def _get_main_variable(poly, var_order):
    """
    Finds the highest-ranking variable (the 'class') in a polynomial.
    Returns None if the polynomial is a constant.
    """
    # Ensure we're working with a Poly object to access its properties
    if not isinstance(poly, Poly):
        poly = Poly(poly)
    for var in var_order:
        if var in poly.gens:
            return var
    return None

def _characteristic_set(poly_list, var_order):
    """
    Computes the characteristic set from a list of polynomials using Wu's method.
    This is a generic, iterative implementation of the triangularization process.

    Args:
        poly_list (list): A list of hypothesis polynomials.
        var_order (list): The variable ordering in descending rank.

    Returns:
        list: A triangularized list of polynomials (the characteristic set).
    """
    # Use a working set of Poly objects for robust manipulation
    working_set = {Poly(p) for p in poly_list}
    print("--- Starting Characteristic Set Computation ---")
    print(f"Initial polynomials: {[p.as_expr() for p in working_set]}")

    reduction_made = True
    while reduction_made:
        reduction_made = False
        # Use a list comprehension to iterate over a copy, allowing safe removal
        polynomials_to_check = list(working_set)
        
        for i in range(len(polynomials_to_check)):
            for j in range(i + 1, len(polynomials_to_check)):
                p1 = polynomials_to_check[i]
                p2 = polynomials_to_check[j]

                main_var1 = _get_main_variable(p1, var_order)
                main_var2 = _get_main_variable(p2, var_order)

                # Skip if either is a constant or they don't share a main variable
                if not main_var1 or not main_var2 or main_var1 != main_var2:
                    continue

                # We have a pair with the same main variable. A reduction is necessary.
                print(f"\nCollision found on variable '{main_var1}':")
                print(f"  P1: {p1.as_expr()}")
                print(f"  P2: {p2.as_expr()}")

                # Decide which polynomial to reduce based on degree
                if degree(p1, main_var1) >= degree(p2, main_var1):
                    poly_to_reduce, reducer = p1, p2
                else:
                    poly_to_reduce, reducer = p2, p1
                
                print(f"Reducing '{poly_to_reduce.as_expr()}' by '{reducer.as_expr()}'...")
                
                # Perform the pseudo-division
                remainder = prem(poly_to_reduce, reducer, main_var1)

                # The remainder might have factors that can be simplified
                remainder = Poly(factor(remainder.as_expr()))

                print(f"Pseudo-remainder: {remainder.as_expr()}")

                # Replace the original polynomial with the remainder
                working_set.remove(poly_to_reduce)
                
                if not remainder.is_zero:
                    working_set.add(remainder)
                
                # Signal that a reduction was made and restart the process
                reduction_made = True
                break  # Exit the inner loop
            if reduction_made:
                break  # Exit the outer loop

    print("\n--- Characteristic Set Computation Complete ---")
    
    # Sort the final set for deterministic output, highest class first
    sorted_char_set = sorted(
        list(working_set),
        key=lambda p: var_order.index(_get_main_variable(p, var_order)) if _get_main_variable(p, var_order) else float('inf')
    )
    
    print(f"Final Characteristic Set: {[p.as_expr() for p in sorted_char_set]}")
    return [p.as_expr() for p in sorted_char_set]


def _verify_conclusions(conclusions, char_set, var_order):
    """
    Verifies a list of conclusions against the characteristic set by computing
    the successive pseudo-remainder for each.

    Args:
        conclusions (list): A list of conclusion polynomials.
        char_set (list): The characteristic set.
        var_order (list): The variable ordering.

    Returns:
        dict: A dictionary mapping each conclusion to its final remainder.
    """
    results = {}

    for i, conclusion in enumerate(conclusions):
        print(f"\nVerifying Conclusion #{i+1}: {conclusion} = 0")
        final_rem = Poly(conclusion, *var_order) # Initialize with explicit generators
        
        # Sequentially reduce the conclusion by each polynomial in the characteristic set
        for poly_expr in char_set:
            poly = Poly(poly_expr, *var_order) # Ensure poly also has explicit generators
            main_var = _get_main_variable(poly, var_order)
            
            # Only reduce if the variable exists in the remainder
            if main_var and main_var in final_rem.gens:
                print(f"  Reducing '{final_rem.as_expr()}' by '{poly.as_expr()}' w.r.t '{main_var}'")
                final_rem = prem(final_rem, poly, main_var)
                # Keep it simple
                # Pass generators explicitly after factoring
                final_rem = Poly(factor(final_rem.as_expr()), *var_order) 
                print(f"  New remainder: {final_rem.as_expr()}")

        results[conclusion] = final_rem.as_expr()
        
    print("\n--- Verification Complete ---")
    return results

def _identify_independent_variables(char_set_expressions, all_variables_in_order):
    """
    Identifies the independent variables from a characteristic set and the
    complete list of variables in their defined order.

    Args:
        char_set_expressions (list): A list of SymPy expressions representing the
                                     characteristic set.
        all_variables_in_order (list): The original ordered list of all variables
                                       in the system.

    Returns:
        list: A list of independent variables.
    """
    main_variables_in_char_set = set()
    
    for poly_expr in char_set_expressions:
        # Convert the expression to a Poly object for accurate main variable identification
        poly = Poly(poly_expr, *all_variables_in_order)
        main_var = _get_main_variable(poly, all_variables_in_order)
        if main_var:
            main_variables_in_char_set.add(main_var)
            
    independent_variables = [
        var for var in all_variables_in_order 
        if var not in main_variables_in_char_set
    ]
    
    return independent_variables

def _natural_sort_key(s):
    """
    Provides a key for sorting strings in a 'natural' order (e.g., a10 > a2).
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', str(s))]

def _determine_variable_order(eqs, concs):
    """
    Automatically determines the variable ordering from all symbols present
    in the hypotheses and conclusions using a natural sort.
    """
    all_symbols = set()
    for p in eqs + concs:
        all_symbols.update(p.free_symbols)
    # Sort in reverse for descending rank (e.g., z, y, x)
    return sorted(list(all_symbols), key=_natural_sort_key, reverse=False)

def prover_func(constraints, conclusions, timeout = 100):

    print(constraints)
    print(conclusions)

    #push all equations to the left side and remove inequalities
    eqs = [sympy.expand_mul(c["equation"].lhs-c["equation"].rhs) for c in constraints if isinstance(c["equation"], sympy.core.relational.Equality)]
    concs = [sympy.expand_mul(c["equation"].lhs-c["equation"].rhs) for c in conclusions if isinstance(c["equation"], sympy.core.relational.Equality)]
    

    print(eqs)
    print(concs)

    if len(eqs) + len(concs) < len(constraints) + len(conclusions):
        print("[WARNING] Inequalities are being ignored by prover.")

    #get all variables and decide variable order
    variable_order = _determine_variable_order(eqs, concs)

    # 3. Phase 1: Compute the Characteristic Set
    print("[INFO] Starting Characteristic Set computation.")
    char_set = _characteristic_set(eqs, variable_order)

    # 4. Identify Independent Variables
    independent_vars = _identify_independent_variables(char_set, variable_order)
    print(f"[INFO]  Identified Independent Variables: {independent_vars}.")

    # 4. Phase 2: Verify the conclusions
    print("[INFO] Starting Verification of Conclusions.")
    final_remainders = _verify_conclusions(concs, char_set, variable_order)

    # 5. Interpret the results
    i = 0
    for conc, rem in final_remainders.items():
        print(f"Conclusion: {conc} = 0")
        print(f"Final Remainder: {rem}")
        if rem == 0:
            print("Verdict: PROVEN TRUE. The proof statement is a consequence of the constraints.\n")
            conclusions[i]["Proven"] = True
        else:
            print(f"Verdict: NOT PROVEN. The remainder is non-zero.\n")
            conclusions[i]["Proven"] = False
        i+=1
