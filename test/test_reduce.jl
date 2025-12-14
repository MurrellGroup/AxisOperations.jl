@testset "Reduce" begin
    @testset "basic correctness" begin
        A = reshape(collect(1:48), 2, 3, 4, 2)
        r = Reduce(sum; dims=(2, 3))
        y = r(A)
        @test y == sum(A; dims=(2, 3))
    end

    @testset "dims=() no-op (preserve wrapper)" begin
        A = reshape(collect(1:24), 2, 3, 4)
        x = PermutedDimsArray(A, (2, 3, 1))
        r = Reduce(sum; dims=())
        @test r(A) === A
        @test r(x) === x
    end

    @testset "PermutedDimsArray unwrap when only reduced dims moved" begin
        A = reshape(collect(1:48), 2, 3, 4, 2)
        p = (1, 3, 2, 4)
        x = PermutedDimsArray(A, p)

        r = Reduce(sum; dims=(2, 3))
        y = r(x)
        @test y == sum(x; dims=(2, 3))
        @test !(y isa PermutedDimsArray)
    end

    @testset "PermutedDimsArray no unwrap when non-reduced dims moved" begin
        A = reshape(collect(1:48), 2, 3, 4, 2)
        p = (2, 1, 3, 4)
        x = PermutedDimsArray(A, p)

        r = Reduce(sum; dims=(4,))
        y = r(x)
        @test y == sum(x; dims=(4,))
        @test !(y isa PermutedDimsArray)
    end

    @testset "errors" begin
        A = reshape(collect(1:24), 2, 3, 4)
        @test_throws ArgumentError Reduce(sum; dims=(-1,))  # constructor checks non-negative
        # Out-of-range dims are allowed in Base (treated as no-op).
        @test Reduce(sum; dims=(5,))(A) == sum(A; dims=(5,))
    end
end


