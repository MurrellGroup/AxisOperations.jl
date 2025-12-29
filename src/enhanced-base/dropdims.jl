@constprop function dropdims(
    x::AbstractArray{<:Any,N}; dims
) where N
    dims′ = dims isa Int ? (dims,) : dims
    ops = ntuple(i -> i in dims′ ? Squeeze() : Keep(), N)
    return reshape(x, ops)
end
