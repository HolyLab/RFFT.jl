import RFFT
using Test, FFTW, LinearAlgebra

for region in (1:2, 1, 2)
    for sz in ((5,6), (6,5))
        pair = RFFT.RCpair(Float64, sz, region)
        r = @inferred(real(pair))
        c = @inferred(complex(pair))
        b = rand(eltype(r), size(r))
        copyto!(r, b)
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
