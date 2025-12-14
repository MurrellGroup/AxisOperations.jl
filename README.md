# AxisOperations

[![Build Status](https://github.com/MurrellGroup/AxisOperations.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/AxisOperations.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/AxisOperations.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/AxisOperations.jl)

AxisOperations.jl is a library providing an intermediate layer for array axis operations with compile-time awareness of wrappers, and the structure of operations.

## Motivation

Julia infamously struggles with "double wrappers", particularly on GPUs, triggering generic scalar indexing fallbacks. This can make users wary when working with lazy wrappers. In the case of `reshape`, the *structure* of the new shape relative to the old shape is completely neglected and an array can for example become a reshape of a view:

```julia
julia> x = rand(3, 4, 2);

julia> reshape(x, size(x, 1), :) isa Array # same type, no copy
true

julia> y = view(x, 1:2, :, :);

julia> reshape(y, size(y, 1), :) isa Base.ReshapedArray
true
```

We may use `size(y, 1)` in our reshape, but despite preserving the first dimension (the one dimension only partially sliced) it evaluates to an integer at runtime, and Julia has no way of knowing that it represents preserving the size of the first dimension. The size could in theory be constant-propagated [if the size wasn't dynamic](https://github.com/JuliaArrays/FixedSizeArrays.jl), but even then the size alone may not be enough.

AxisOperations.jl provides a way to be explicit about the structure of the reshape operation at compile-time (or code-write-time), enabling optimizations in the form of "rewrapping" the array.

```julia
julia> using AxisOperations

julia> reshape(y, Keep(1), :) isa SubArray
true
```

As a more complex example, we can "split" the first dimension into two:

```julia
julia> x = rand(12, 2);

julia> y = view(x, 1:6, :);

julia> reshape(y, Split(1, (2, :)), :)
2×3×2 view(::Array{Float64, 3}, :, 1:3, :) with eltype Float64:
[:, :, 1] =
 0.509548  0.0287457  0.697172
 0.442275  0.767883   0.170847

[:, :, 2] =
 0.717791  0.201782  0.00850527
 0.265492  0.401664  0.080724
```

The view gets unwrapped, the original array gets reshaped, and the viewed first dimension carries over to the second dimension.

## Features

- `..` (from [EllipsisNotation.jl](https://github.com/SciML/EllipsisNotation.jl)) is replaced with `Keep(..)` when passed to `reshape`.
- `:` can be used like normal, but under the hood it gets replaced by `Merge(..)` when passed to `reshape`.

## Limitations

- Direct arguments of reshape can not be integers when an axis operation is present.
- `..` and `:` alone won't use AxisOperations.jl, as defining such methods would be type piracy. In these cases, `Keep(..)` and `Merge(..)` should be used instead.
