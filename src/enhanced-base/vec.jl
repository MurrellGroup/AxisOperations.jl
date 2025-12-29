"""
    vec(x)

Flatten the array `x` into a vector.

```jldoctest
julia> x = rand(2, 3, 4);

julia> vec(x)
24-element Vector{Float64}:
 0.560475
 0.188602
```
"""
vec(x) = reshape(x, Merge(..))
