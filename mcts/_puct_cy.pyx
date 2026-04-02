# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython-accelerated PUCT selection for MCTS."""

from libc.math cimport sqrt

cdef double PUCT_C = 0.8
cdef double FPU_REDUCTION = 0.25


def puct_select(node, double c=PUCT_C, double fpu=0.0):
    """Select child with highest PUCT score. Returns action index.

    *fpu* is the fallback Q-value for unvisited children.  Once siblings
    have visits, FPU is dynamically computed (KataGo-style).
    """
    cdef:
        int n = node.n
        list actions = node.actions
        list priors = node.priors
        list visits = node.visits
        list values = node.values
        int visit_count = <int>node.visit_count
        double c_sqrt = c * sqrt(<double>visit_count)
        double best = -1e30
        double q, s, p
        int best_a = -1
        int vc, i
        double total_val, mass, mean_q, nn_value, w, fpu_base

    # Dynamic FPU: once children have visits, track observed Q
    if visit_count > 0:
        total_val = 0.0
        mass = 0.0
        for i in range(n):
            vc = <int>visits[i]
            if vc > 0:
                total_val += <double>values[i]
                mass += <double>priors[i]
        if mass > 0.0:
            mean_q = total_val / visit_count
            nn_value = fpu + FPU_REDUCTION
            w = mass if mass < 1.0 else 1.0
            fpu_base = w * mean_q + (1.0 - w) * nn_value
            fpu = fpu_base - FPU_REDUCTION * sqrt(w)

    if node._has_terminal:
        terminals = node.terminals
        term_vals = node.term_vals
        for i in range(n):
            vc = <int>visits[i]
            if <bint>terminals[i]:
                q = <double>term_vals[i]
            elif vc > 0:
                q = <double>values[i] / vc
            else:
                q = fpu
            p = <double>priors[i]
            s = q + c_sqrt * p / (1 + vc)
            if s > best:
                best = s
                best_a = <int>actions[i]
    else:
        for i in range(n):
            vc = <int>visits[i]
            if vc > 0:
                q = <double>values[i] / vc
            else:
                q = fpu
            p = <double>priors[i]
            s = q + c_sqrt * p / (1 + vc)
            if s > best:
                best = s
                best_a = <int>actions[i]
    return best_a
