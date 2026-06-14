"""Metal-ZK: zero-knowledge / lattice cryptography kernel benchmark on Apple Silicon Metal.

A sibling benchmark to Metal-Sci. Same evolutionary harness, but the
correctness gate is bit-exact integer equality and the roofline anchors
on int64-mul throughput instead of FP32 GFLOPS.
"""
