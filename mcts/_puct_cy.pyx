# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython-accelerated PUCT selection for MCTS."""

from libc.math cimport sqrt

cdef double PUCT_C = 0.8


def puct_select(node, double c=PUCT_C, double fpu=0.0):
    """Select child with highest PUCT score. Returns action index.

    *fpu* is the Q-value for unvisited children (First Play Urgency).
    """
    cdef:
        int n = node.n
        list actions = node.actions
        list priors = node.priors
        list visits = node.visits
        list values = node.values
        double c_sqrt = c * sqrt(<double>node.visit_count)
        double best = -1e30
        double q, s, p
        int best_a = -1
        int vc, i

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
