function _pda_reshape_codegen(T, N::Int, M::Int, op_types::Core.SimpleVector, perm, iperm)
    fallback() = :(invoke(r, Tuple{AbstractArray{$T,$N}}, x))

    _ops_has_ellipsis(op_types) && return fallback()

    total_in = _ops_total_in(op_types)
    (total_in === nothing || total_in != N) && return fallback()

    starts = Int[]
    outcounts = Int[]
    parent_ops = Any[]

    in_dim = 0

    function push_group!(start::Int, outcount::Int, op_expr)
        push!(starts, start)
        push!(outcounts, outcount)
        push!(parent_ops, op_expr)
        return nothing
    end

    for (k, opT) in enumerate(op_types)
        n_in = ndims_in(opT)
        m_out = ndims_out(opT)

        if opT <: Keep
            for _ in 1:n_in
                in_dim += 1
                push_group!(perm[in_dim], 1, :(Keep(1)))
            end

        elseif opT <: Merge
            if n_in == 0
                push_group!(typemax(Int), 1, :(ops[$k]))
            elseif n_in == 1
                in_dim += 1
                push_group!(perm[in_dim], 1, :(Keep(1)))
            else
                first = perm[in_dim + 1]
                for j in 2:n_in
                    perm[in_dim + j] == first + (j - 1) || return fallback()
                end
                in_dim += n_in
                push_group!(first, 1, :(ops[$k]))
            end

        elseif opT <: Split
            n_in == 0 && return fallback()
            first = perm[in_dim + 1]
            for j in 2:n_in
                perm[in_dim + j] == first + (j - 1) || return fallback()
            end
            in_dim += n_in
            push_group!(first, m_out, :(ops[$k]))

        elseif opT <: Resqueeze
            if n_in == 0
                push_group!(typemax(Int), m_out, :(ops[$k]))
            elseif m_out == 0
                for _ in 1:n_in
                    in_dim += 1
                    push_group!(perm[in_dim], 0, :(Squeeze(1)))
                end
            else
                return fallback()
            end
        else
            return fallback()
        end
    end

    in_dim == N || return fallback()

    order = sortperm(starts)

    first_outpos = fill(0, length(outcounts))
    out_cursor = 0
    for gi in order
        oc = outcounts[gi]
        if oc > 0
            first_outpos[gi] = out_cursor + 1
            out_cursor += oc
        end
    end

    new_perm = Int[]
    for gi in 1:length(outcounts)
        oc = outcounts[gi]
        if oc > 0
            start = first_outpos[gi]
            for j in 0:(oc - 1)
                push!(new_perm, start + j)
            end
        end
    end

    parent_ops_sorted = Any[parent_ops[i] for i in order]
    new_perm_tuple = Tuple(new_perm)

    return quote
        ops = r.ops
        parent_ops = $(Expr(:tuple, parent_ops_sorted...))
        parent_r = resolve(parent_ops, Val($N))
        rp = parent_r(parent(x))
        PermutedDimsArray(rp, $new_perm_tuple)
    end
end

@generated function (r::Reshape{OpsT,N,M})(
    x::PermutedDimsArray{T,N,perm,iperm,P},
) where {OpsT,N,M,T,perm,iperm,P}
    _pda_reshape_codegen(T, N, M, OpsT.parameters, perm, iperm)
end
