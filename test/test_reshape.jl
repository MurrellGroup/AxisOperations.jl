@testset "Reshape" begin

    @testset "Equivalence" begin
        A = reshape(collect(1:24), 4, 3, 2)

        @test reshape(A, Keep(3)) == A
        @test reshape(A, Keep(..)) == A
        @test reshape(A, Merge(2), Keep(1)) == reshape(A, 12, 2)
        @test reshape(A, Split(1, (2, 2)), Keep(2)) == reshape(A, 2, 2, 3, 2)
        @test reshape(A, Split(1, (2, :)), Keep(2)) == reshape(A, 2, 2, 3, 2)
        @test reshape(A, Split(1, (1, 4)), Keep(..)) == reshape(A, 1, 4, 3, 2)
    end

    @testset "Colon notation" begin
        A = reshape(collect(1:24), 4, 3, 2)
        @test reshape(A, Keep(1), :) == reshape(A, 4, :)
    end

    @testset "Ellipsis notation" begin
        A = reshape(collect(1:24), 4, 3, 2)
        @test reshape(A, Split(1, (2, :)), ..) == reshape(A, 2, :, 3, 2)
    end

    @testset "Resqueeze" begin
        A = reshape(collect(1:24), 4, 3, 2)
        x = reshape(A, 1, 4, 3, 2)

        y = reshape(x, Squeeze(1), Keep(..))
        @test y == A
        z = reshape(A, Unsqueeze(1), Keep(..))
        @test z == x
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

        E = reshape(collect(1:32), 8, 2, 2)
        x5 = view(E, 1:4, :, :)
        y5_int_first = reshape(x5, Split(1, (2, :)), Keep(..))
        y5_colon_first = reshape(x5, Split(1, (:, 2)), Keep(..))
        @test y5_int_first == y5_colon_first
        @test y5_colon_first == reshape(x5, 2, 2, 2, 2)
        @test y5_colon_first isa SubArray
        @test _shares_storage(y5_colon_first, parent(x5))
    end

    @testset "PermutedDimsArray reshape wrapper elision" begin
        A = reshape(collect(1:24), 2, 3, 4)
        x = PermutedDimsArray(A, (3, 1, 2))
        @test reshape(x, Keep(3)) == x
        @test reshape(x, Keep(), :) == PermutedDimsArray(reshape(A, :, 4), (2, 1))
        @test _shares_storage(A, x)
    end

    @testset "ReinterpretArray reshape wrapper elision" begin
        data = UInt32[0x00010002, 0x00030004, 0x00050006, 0x00070008, 0x00090010, 0x00110012]

        @testset "false mode (scaled first dim)" begin
            M = reshape(data, 3, 2)
            x = reinterpret(UInt16, M)
            @test size(x) == (6, 2)

            @test reshape(x, Keep(..)) === x
            @test reshape(x, Keep(2)) === x

            y_merge = reshape(x, Merge(..))
            @test y_merge isa Base.ReinterpretArray
            @test size(y_merge) == (12,)
            @test collect(y_merge) == vec(collect(x))

            y_km = reshape(x, Keep(), Merge(..))
            @test y_km isa Base.ReinterpretArray
            @test size(y_km) == (6, 2)
            @test collect(y_km) == collect(x)

            y_split_opt = reshape(x, Split1(2, :), Keep(..))
            @test y_split_opt isa Base.ReinterpretArray
            @test size(y_split_opt) == (2, 3, 2)
            @test collect(y_split_opt) == reshape(collect(x), 2, 3, 2)

            y_split_fall = reshape(x, Split1(3, :), Keep(..))
            @test y_split_fall isa Base.ReshapedArray
            @test size(y_split_fall) == (3, 2, 2)
            @test collect(y_split_fall) == reshape(collect(x), 3, 2, 2)
        end

        @testset "true mode (ratio as first dim)" begin
            M = reshape(data, 3, 2)
            x = reinterpret(reshape, UInt16, M)
            @test size(x) == (2, 3, 2)

            @test reshape(x, Keep(..)) === x
            @test reshape(x, Keep(3)) === x

            y_merge = reshape(x, Merge(..))
            @test y_merge isa Base.ReinterpretArray
            @test size(y_merge) == (12,)
            @test collect(y_merge) == vec(collect(x))

            y_km = reshape(x, Keep(), Merge(..))
            @test y_km isa Base.ReinterpretArray
            @test size(y_km) == (2, 6)
            @test collect(y_km) == reshape(collect(x), 2, 6)

            y_merge2 = reshape(x, Merge(2), Keep(..))
            @test y_merge2 isa Base.ReinterpretArray
            @test size(y_merge2) == (6, 2)
            @test collect(y_merge2) == reshape(collect(x), 6, 2)
        end

        @testset "value preservation" begin
            M = reshape(data, 3, 2)

            x_false = reinterpret(UInt16, M)
            x_true = reinterpret(reshape, UInt16, M)

            for (name, x, expected_flat) in [
                ("false", x_false, vec(collect(x_false))),
                ("true", x_true, vec(collect(x_true))),
            ]
                y = reshape(x, Merge(..))
                @test collect(y) == expected_flat
            end
        end
    end
end


