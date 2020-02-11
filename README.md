# FastOptimalDamping.jl

Fast computation of optimal damping parameters for linear vibrational systems

This is a Julia package which is used in computing numerical examples for the paper
[Jakovcevic Stor, Slapnicar and Tomljanovic (2020)][JST2020]


[JST2020]: http://xxx "Nevena Jakovcevic Stor, Ivan Slapnicar and Zoran Tomljanovic, 'Fast computation of optimal damping parameters for linear vibrational systems', 2020"

The optimization of viscosities is an order of magnitude faster that the standard approach based on the solution of Lyapunov equation - `O(n^2)` v.s. `O(n^3)` operations, where `n` is the number of masses.
The algorithm uses eigenvalue decomposition of `n-by-n` complex symmetric diagonal-plus-rank one (DPR1) matrix.
Our newly developed algorithm for DPR1 eigenvalue problem uses modified Rayleigh quotient iteration and stores eigenvectors as a Cauchy-like matrix, having complexity of `O(n^2)` operations.
The decomposition is updated as many times as there are dumpers, and it is fast (again `O(n^2)` operations) due to fast multiplication of Cauchy-like matrices.
The algorithm uses multi-threading in a simple and efficient manner.

## Installation

`https://github.com/ivanslapnicar/FastOptimalDamping.jl.git`

## Running tests

Open Julia console, change to the `src/` directory of the package, and run the following commands:

```
include("FastOptimalDamping.jl")
include("SmallExample.jl")
include("LargeExample.jl")
include("HomogeneousExample.jl")
```

On a 12 core Intel i7-8700K, approximate run times are
1 minute for the small example, 3 minutes for the large example and 6 minutes for the homogeneous example.

To run Julia in multi-threading setting in Linux, include the line
```
export JULIA_NUM_THREADS=`nproc`
```
in your `.bashrc` file. For Windows, set the environment variable 
`JULIA_NUM_THREADS` to number of CPUs.
