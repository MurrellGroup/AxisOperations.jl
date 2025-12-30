include("LocalReshape.jl")
include("utils.jl")

"""
    Reshape(ops, ::Val{N})
    Reshape(ops, ::AbstractArray{<:Any,N})

Given a tuple of operations and the input array dimensionality,
construct a `Reshape` object that can be used to reshape arrays.

See also [`Rewrap.reshape`](@ref) and [`Base.reshape`](@ref).
"""
struct Reshape{OpsT,N,M} <: GlobalAxisOp{N,M}
    ops::OpsT
end

@generated function Reshape(ops::OpsT, ::Val{N}) where {OpsT<:Tuple,N}
    raw_op_types = OpsT.parameters
    _, preprocess_result = _preprocess_op_types(raw_op_types)
    
    op_types = Any[]
    op_exprs = Any[]
    
    if preprocess_result === nothing
        for (k, opT) in enumerate(raw_op_types)
            if is_ellipsis(opT)
                push!(op_types, Keep{..})
                push!(op_exprs, :(Keep(..)))
            else
                push!(op_types, opT)
                push!(op_exprs, :(ops[$k]))
            end
        end
    elseif preprocess_result.first === :lone_colon
        idx = preprocess_result.second
        for (k, opT) in enumerate(raw_op_types)
            if k == idx
                push!(op_types, Merge{..})
                push!(op_exprs, :(Merge(..)))
            elseif is_ellipsis(opT)
                push!(op_types, Keep{..})
                push!(op_exprs, :(Keep(..)))
            else
                push!(op_types, opT)
                push!(op_exprs, :(ops[$k]))
            end
        end
    else
        (start, stop) = preprocess_result.second
        split_type = _build_split_type(raw_op_types, start, stop)
        split_expr = _build_split_expr(start, stop)
        
        for (k, opT) in enumerate(raw_op_types)
            if k == start
                push!(op_types, split_type)
                push!(op_exprs, split_expr)
            elseif k > start && k <= stop
                continue
            elseif is_ellipsis(opT)
                push!(op_types, Keep{..})
                push!(op_exprs, :(Keep(..)))
            else
                push!(op_types, opT)
                push!(op_exprs, :(ops[$k]))
            end
        end
    end
    
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
            push!(resolved_op_exprs, :(Split($nin, $(op_exprs[k]).sizes)))
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

Reshape(ops, ::AbstractArray{<:Any,N}) where {N} = Reshape(ops, Val(N))

include("specializations/AbstractArray.jl")
include("specializations/PermutedDimsArray.jl")
include("specializations/SubArray.jl")
include("specializations/ReinterpretArray.jl")
