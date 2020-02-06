# Fast Optimal Damping

Fast computation of optimal damping parameters for linear vibrational systems

This is a Julia package which is used in computing numerical examples for the paper
[Jakovcevic Stor, Slapnicar and Tomljanovic (2020)][JST2020]


[JST2020]: http://xxx "Nevena Jakovcevic Stor, Ivan Slapnicar and Zoran Tomljanovic, 'Fast computation of optimal damping parameters for linear vibrational systems', 2020"

The optimization of viscosities is an order of magnitude faster that the standard approach based on the solution of Lyapunov equation - `O(n^2)` v.s. `O(n^3)` operations, where `n` is the number of masses.
The algorithm uses eigenvalue decomposition of `n-by-n` complex symmetric diagonal-plus-rank one (DPR1) matrix.
Our newly developed algorithm for DPR1 eigenvalue problem uses modified Rayleigh quotient iteration and stores eigenvectors as a Cauchy-like matrix, having complexity of `O(n^2)` operations.
The decomposition is updated as many times as there are dumpers, and it is fast (again `O(n^2)` operations) due to fast multiplication of Cauchy-like matrices.
The algorithm uses multi-threading in a simple and efficient manner.

## Install

`https://github.com/ivanslapnicar/Fast-Optimal-Damping.git`
