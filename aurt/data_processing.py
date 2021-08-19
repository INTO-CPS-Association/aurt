from math import pi
from dataclasses import dataclass
import pandas as pd
import sympy as sp


plot_colors = ['red', 'green', 'blue', 'chocolate', 'crimson', 'fuchsia', 'indigo', 'orange']

JOINT_N = "#"


@dataclass
class ModifiedDH:
    d: list
    a: list
    alpha: list
    q: list
    n_joints: int

    def __init__(self, d, a, alpha, q) -> None:
        self.d = d
        self.a = a
        self.alpha = alpha
        self.q = q
        assert len(self.d) == len(self.a) == len(self.alpha)
        self.set_n_joints()
    
    def set_n_joints(self) -> int:
        self.n_joints = len(self.d)-1


def convert_file_to_mdh(filename):
    if filename[-3:] == "csv":
        df = pd.read_csv(filename)
    df = df.fillna(value='None')
    d = [float(d) if d != 'None' else None for d in df.d]
    a = [float(a) if a != 'None' else None for a in df.a]
    alpha = []
    for alpha_i in df.alpha:
        alpha.append(input_with_pi_to_float(alpha_i))
    # insert the extra joint at index 0
    d.insert(0,float(0))
    a.insert(0,float(0))
    alpha.insert(0,float(0))
    mdh = ModifiedDH(d,a,alpha,None) # TODO: add q to the listing, and remove njoints, remove sp.integer(0)
    q = [sp.Integer(0)] + [sp.symbols(f"q{j}") for j in range(1, mdh.n_joints + 1)]
    mdh.q = q
    return mdh

def input_with_pi_to_float(input):
    if input == 'None':
        return None
    elif isinstance(input, str) and "pi" in input:
        input = input.split("/")
        # Case: no / in input, i.e. either pi or a number, or -pi
        if len(input) == 1: # this means no / is in input
            if "pi" == input[0]:
                return pi
            elif "-pi" == input[0]:
                return -pi
            else:
                return float(input)
        # Case: / in input, either pi/2, pi/4, -pi/2, -pi/8, or number/number
        elif len(input) == 2:
            if "-pi" == input[0]:
                return -pi/float(input[1])
            elif "pi" == input[0]:
                return pi/float(input[1])
        else:
            print(f"Whoops, len of input is greater than 2: {len(input)}")
    else:
        return float(input)
