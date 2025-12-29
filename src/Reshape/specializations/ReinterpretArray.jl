function _reinterp_reshape_codegen(T, N::Int, M::Int, op_types::Core.SimpleVector, S, check::Bool)
    fallback() = :(invoke(r, Tuple{AbstractArray{$T,$N}}, x))

    _ops_has_ellipsis(op_types) && return fallback()

    total_in = _ops_total_in(op_types)
    (total_in === nothing || total_in != N) && return fallback()

    all(op -> op <: Keep, op_types) && return :x

    all_merge = length(op_types) == 1 && op_types[1] <: Merge
    all_merge && return :(reinterpret($T, Base.reshape(parent(x), Merge(..))))

    parent_N = check ? N - 1 : N

    function build_parent_ops(skip_first::Bool, first_op_override=nothing)
        parent_ops = Any[]
        for (k, opT) in enumerate(op_types)
            if k == 1 && skip_first
                first_op_override !== nothing && push!(parent_ops, first_op_override)
                continue
            end
            if opT <: Keep
                push!(parent_ops, :(Keep($(ndims_in(opT)))))
            elseif opT <: Merge
                push!(parent_ops, :(Merge($(ndims_in(opT)))))
            elseif opT <: Split
                push!(parent_ops, :(ops[$k]))
            elseif opT <: Resqueeze
                push!(parent_ops, :(ops[$k]))
            else
                return nothing
            end
        end
        return Expr(:tuple, parent_ops...)
    end

    if check
        first_op = op_types[1]
        n_in_first = ndims_in(first_op)

        if first_op <: Keep && n_in_first == 1
            parent_ops_tuple = build_parent_ops(true)
            parent_ops_tuple === nothing && return fallback()
            return quote
                ops = r.ops
                parent_ops = $parent_ops_tuple
                parent_r = resolve(parent_ops, Val($parent_N))
                reinterpret(Base.reshape, $T, parent_r(parent(x)))
            end

        elseif first_op <: Merge && n_in_first >= 1
            merge_count = n_in_first - 1
            parent_ops_tuple = build_parent_ops(true, :(Merge($merge_count)))
            parent_ops_tuple === nothing && return fallback()
            return quote
                ops = r.ops
                parent_ops = $parent_ops_tuple
                parent_r = resolve(parent_ops, Val($parent_N))
                reinterpret($T, parent_r(parent(x)))
            end
        else
            return fallback()
        end
    else
        first_op = op_types[1]
        n_in_first = ndims_in(first_op)

        if (first_op <: Keep && n_in_first == 1) || (first_op <: Merge && n_in_first >= 1)
            parent_ops_tuple = build_parent_ops(false)
            parent_ops_tuple === nothing && return fallback()
            return quote
                ops = r.ops
                parent_ops = $parent_ops_tuple
                parent_r = resolve(parent_ops, Val($parent_N))
                reinterpret($T, parent_r(parent(x)))
            end

        elseif first_op <: Split && n_in_first == 1
            ratio = sizeof(S) ÷ sizeof(T)
            sizes_type = first_op.parameters[3]
            sizes_type <: Tuple || return fallback()
            length(sizes_type.parameters) >= 1 || return fallback()
            first_size_type = sizes_type.parameters[1]
            first_size_type <: Int || return fallback()

            parent_ops_tuple = build_parent_ops(true, :(Split(1, (first_size ÷ $ratio, ops[1].sizes[2:end]...))))
            parent_ops_tuple === nothing && return fallback()
            return quote
                ops = r.ops
                first_size = ops[1].sizes[1]
                if first_size % $ratio == 0
                    parent_ops = $parent_ops_tuple
                    parent_r = resolve(parent_ops, Val($parent_N))
                    reinterpret($T, parent_r(parent(x)))
                else
                    invoke(r, Tuple{AbstractArray{$T,$N}}, x)
                end
            end
        else
            return fallback()
        end
    end
end

@generated function (r::Reshape{OpsT,N,M})(
    x::Base.ReinterpretArray{T,N,S,A,check},
) where {OpsT,N,M,T,S,A,check}
    _reinterp_reshape_codegen(T, N, M, OpsT.parameters, S, check)
end
