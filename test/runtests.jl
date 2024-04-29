import RFFT
using Test, FFTW, LinearAlgebra

@testset "RFFT.jl" begin
    for dims in (1:2, 1, 2)
        for sz in ((5,6), (6,5))
            pair = RFFT.RCpair{Float64}(undef, sz, dims)
            r = @inferred(real(pair))
            c = @inferred(complex(pair))
            b = rand(eltype(r), size(r))
            pair = RFFT.RCpair(b, dims)
            copyto!(r, b)
            copy!(pair, c) # for coverage
            RFFT.rfft!(pair)
            RFFT.irfft!(pair)
            @test r ≈ b
            pfwd = RFFT.plan_rfft!(pair)
            pinv = RFFT.plan_irfft!(pair)
            pinv(pfwd(pair))
            @test r ≈ b

            pair2 = copy(pair)
            @test real(pair2) == real(pair)
            @test complex(pair2) == complex(pair)
        end
    end
end
