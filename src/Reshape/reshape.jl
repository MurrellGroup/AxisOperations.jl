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
