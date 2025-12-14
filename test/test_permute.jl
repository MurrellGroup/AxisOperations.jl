@testset "Permute" begin
    @testset "composition / wrapper elision" begin
        A = reshape(collect(1:24), 2, 3, 4)
        x = PermutedDimsArray(A, (2, 3, 1))

        # Composition becomes identity => drop PermutedDimsArray wrapper
        y = Permute((3, 1, 2))(x)
        @test y === A

        # Composition stays non-trivial => fuse to a single PermutedDimsArray(parent, perm)
        z = Permute((1, 3, 2))(x)
        @test z isa PermutedDimsArray
        @test parent(z) === A
        @test z == permutedims(x, (1, 3, 2))
    end

    @testset "transpose/adjoint" begin
        M = reshape(collect(1:6), 2, 3)
        t = transpose(M)
        @test Permute((2, 1))(t) === M

        R = reshape(Float64.(1:6), 2, 3)
        a = adjoint(R)
        @test Permute((2, 1))(a) === R
    end

    @testset "errors" begin
        A = reshape(collect(1:24), 2, 3, 4)
        @test_throws ArgumentError Permute((1, 1, 2))(A)
    end
end




