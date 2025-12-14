using AxisOperations
using LinearAlgebra
using Test

include("utils.jl")

@testset "AxisOperations.jl" begin
    include("test_reshape.jl")
    include("test_permute.jl")
    include("test_repeat.jl")
    include("test_reduce.jl")
end
