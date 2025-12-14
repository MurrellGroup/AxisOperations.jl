function _shares_storage(a, b)
    # Peel wrapper parents where it actually unwraps to something different.
    while true
        pa = (Base.hasmethod(parent, Tuple{typeof(a)}) ? parent(a) : a)
        pb = (Base.hasmethod(parent, Tuple{typeof(b)}) ? parent(b) : b)
        (pa === a && pb === b) && break
        a = pa
        b = pb
    end

    # `pointer` equality is sufficient but not necessary (views can alias with different offsets),
    # so prefer `mightalias` and use pointer equality only as a fast positive.
    if a isa StridedArray && b isa StridedArray
        Base.mightalias(a, b) && return true
        (isempty(a) || isempty(b)) && return false
        return pointer(a) == pointer(b)
    end

    return Base.mightalias(a, b)
end
