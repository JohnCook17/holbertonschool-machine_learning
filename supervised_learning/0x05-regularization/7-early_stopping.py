#!/usr/bin/env python3
"""A algorithm for stopping early"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """logic for determining if we should stop early"""
    if opt_cost - cost <= threshold:
        count += 1
        if count < patience:
            return False, count
        else:
            return True, count
    else:
        return False, 0
