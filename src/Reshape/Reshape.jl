include("LocalReshape.jl")
include("utils.jl")

struct Reshape{OpsT,N,M} <: GlobalAxisOp{N,M}
    ops::OpsT
end

@generated function resolve(ops::OpsT, ::Val{N}) where {OpsT<:Tuple,N}
    op_types = OpsT.parameters
    _ops_has_ellipsis(op_types) || nothing

    ellipsis_seen = false
    fixed_in = 0
    has_ellipsis_out = false

    for opT in op_types
        n_in = ndims_in(opT)
        if is_ellipsis(n_in)
            ellipsis_seen && throw(ArgumentError("At most one op can have Ellipsis input"))
            ellipsis_seen = true
        else
            fixed_in += n_in
        end
        has_ellipsis_out |= is_ellipsis(ndims_out(opT))
    end

    ellipsis_n = if ellipsis_seen
        n = N - fixed_in
        n >= 0 || throw(ArgumentError("Ops need $fixed_in dims but array has $N"))
        n
    else
        has_ellipsis_out && throw(ArgumentError("Ellipsis output requires an Ellipsis input"))
        fixed_in == N || throw(ArgumentError("Ops consume $fixed_in dims but array has $N"))
        0
    end

    resolved_in(opT) = is_ellipsis(ndims_in(opT)) ? ellipsis_n : ndims_in(opT)
    resolved_out(opT) = is_ellipsis(ndims_out(opT)) ? ellipsis_n : ndims_out(opT)

    consumed = 0
    produced = 0

    resolved_op_exprs = Any[]
    resolved_op_types = Any[]

    for (k, opT) in enumerate(op_types)
        nin = resolved_in(opT)
        mout = resolved_out(opT)
        consumed += nin
        produced += mout

        if opT <: Keep
            push!(resolved_op_exprs, :(Keep($nin)))
            push!(resolved_op_types, Keep{nin})

        elseif opT <: Merge
            push!(resolved_op_exprs, :(Merge($nin)))
            push!(resolved_op_types, Merge{nin})

        elseif opT <: Split
            T0 = opT.parameters[3]
            M0 = opT.parameters[2]
            mout == M0 || throw(ArgumentError("Split output rank cannot be ellipsis-resolved"))
            push!(resolved_op_exprs, :(Split($nin, ops[$k].sizes)))
            push!(resolved_op_types, Split{nin, M0, T0})

        elseif opT <: Resqueeze
            push!(resolved_op_exprs, :(Resqueeze($nin => $mout)))
            push!(resolved_op_types, Resqueeze{nin, mout})

        else
            throw(ArgumentError("Unknown reshape op type: $opT"))
        end
    end

    consumed == N || throw(ArgumentError("Ops consume $consumed dims but array has $N"))

    resolved_opsT = Tuple{resolved_op_types...}
    resolved_ops_expr = Expr(:tuple, resolved_op_exprs...)

    return :(Reshape{$resolved_opsT,$N,$produced}($resolved_ops_expr))
end

resolve(ops, ::AbstractArray{<:Any,N}) where {N} = resolve(ops, Val(N))

include("specializations/AbstractArray.jl")
include("specializations/PermutedDimsArray.jl")
include("specializations/SubArray.jl")
include("specializations/ReinterpretArray.jl")

"""
    reshape(x, ops::Union{LocalReshape,Colon,EllipsisNotation.Ellipsis}...)

Reshape the array `x` using the given operations.

!!! note
    `ops` *must* contain at least one `LocalReshape`.

```jldoctest
julia> x = rand(3, 5, 2);

julia> x′ = reshape(x, Keep(), :);

julia> size(x′)
(3, 10)

julia> y′ = rand(2, 3) * x′; # project from 3 to 2

julia> size(y′)
(2, 10)

julia> y = reshape(y′, Keep(), Split(1, size(x)[2:end]));

julia> size(y)
(2, 5, 2)
```
"""
Base.reshape

@constprop function Base.reshape(
    x::AbstractArray{<:Any,N}, ops::Tuple{LocalReshape,Vararg{LocalReshape}}
) where N
    r = resolve(ops, Val(N))
    r(x)
end

@constprop function Base.reshape(
    x::AbstractArray,
    ops::Union{
        Tuple{ColonOrEllipsis,LocalReshape,Vararg{LocalReshape}},
        Tuple{LocalReshape,Vararg{Union{LocalReshape,ColonOrEllipsis}}}
    }
)
    count(op -> op isa ColonOrEllipsis, ops) > 1 && throw(ArgumentError("At most one Colon or Ellipsis is allowed"))
    ops′ = map(ops) do op
        if op isa Colon
            Merge(..)
        elseif op isa Ellipsis
            Keep(..)
        else
            op
        end
    end
    reshape(x, ops′)
end

@constprop function Base.reshape(
    x::AbstractArray, op1::LocalReshape, ops::Union{LocalReshape,ColonOrEllipsis}...
)
    return reshape(x, (op1, ops...))
end

@constprop function Base.reshape(
    x::AbstractArray, op1::ColonOrEllipsis, op2::LocalReshape, ops::LocalReshape...
)
    return reshape(x, (op1, op2, ops...))
end
