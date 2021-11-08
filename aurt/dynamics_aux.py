
import sympy as sp
from itertools import chain

def list_2D_to_sympy_vector(my_2D_list):
        """Flattens 2D list to 1D list and converts 1D list to sympy.Matrix() object."""
        return sp.Matrix(list(chain.from_iterable(my_2D_list)))


def sym_mat_to_subs(sym_mats, num_mats):
    subs = {}

    for s_mat, n_mat in zip(sym_mats, num_mats):
        subs = {**subs, **{s: v for s, v in zip(s_mat, n_mat) if s != 0}}

    return subs

def number_of_elements_in_nested_list(element):
    count = 0
    if isinstance(element, list):
        for each_element in element:
            count += number_of_elements_in_nested_list(each_element)
    else:
        count += 1
    return count