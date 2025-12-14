@testset "Reshape" begin
    @testset "shape correctness vs Base.reshape" begin
        A = reshape(collect(1:24), 4, 3, 2)

        y1 = reshape(A, Keep(3))
        @test y1 == A

        y2 = reshape(A, Merge(2), Keep(1))
        @test size(y2) == (12, 2)
        @test y2 == reshape(A, 12, 2)

        y3 = reshape(A, Split(1, (2, 2)), Keep(2))
        @test size(y3) == (2, 2, 3, 2)
        @test y3 == reshape(A, 2, 2, 3, 2)

        y4 = reshape(A, Split(1, (2, :)), Keep(2))
        @test size(y4) == (2, 2, 3, 2)
        @test y4 == reshape(A, 2, 2, 3, 2)

        y5 = reshape(A, Split(1, (1, 4)), Keep(..))
        @test size(y5) == (1, 4, 3, 2)
        @test y5 == reshape(A, 1, 4, 3, 2)
    end

    @testset "Resqueeze" begin
        A = reshape(collect(1:24), 4, 3, 2)
        x = reshape(A, 1, 4, 3, 2)

        y = reshape(x, Squeeze(1), Keep(..))
        @test y == reshape(A, 4, 3, 2)

        z = reshape(A, Unsqueeze(1), Keep(..))
        @test size(z) == (1, 4, 3, 2)
        @test z == reshape(A, 1, 4, 3, 2)
    end

    @testset "Split errors" begin
        A = reshape(collect(1:12), 6, 2)
        @test_throws DimensionMismatch reshape(A, Split(1, (4, :)), Keep(..))
        @test_throws ArgumentError reshape(A, Split(1, (0, :)), Keep(..))
        @test_throws ArgumentError reshape(A, Split(1, (2, :, :)), Keep(..))
    end

    @testset "SubArray wrapper elision" begin
        A = reshape(collect(1:24), 4, 3, 2)
        x = view(A, 1:2, :, :)
        y = reshape(x, Split(1, (1, 2)), Keep(..))
        @test y == reshape(x, 1, 2, 3, 2)
        @test y isa SubArray
        @test _shares_storage(y, parent(x))

        B = reshape(collect(1:16), 8, 2)
        x2 = view(B, 2:3, :)
        @test_throws DimensionMismatch reshape(x2, Split(1, (2, :)), Keep(..))

        C = reshape(collect(1:24), 4, 3, 2)
        x3 = view(C, 1:2, :, :)
        y3 = reshape(x3, Split(1, (2,)), Keep(..))
        @test y3 == x3
        @test y3 isa SubArray
        @test _shares_storage(y3, parent(x3))

        D = reshape(collect(1:32), 8, 2, 2)
        x4 = view(D, 1:4, :, :)
        y4 = reshape(x4, Split(1, (2, 1, 2)), Keep(..))
        @test y4 == reshape(x4, 2, 1, 2, 2, 2)
        @test y4 isa SubArray
        @test _shares_storage(y4, parent(x4))
    end

    @testset "PermutedDimsArray reshape wrapper elision" begin
        A = reshape(collect(1:24), 2, 3, 4)
        x = PermutedDimsArray(A, (3, 1, 2))
        @test reshape(x, Keep(3)) == x
        @test reshape(x, Keep(), :) == PermutedDimsArray(reshape(A, :, 4), (2, 1))
        @test _shares_storage(A, x)
    end
end


