import RFFT
using Base.Test

for region in (1:2, 1, 2)
    for sz in ((5,6), (6,5))
        pair = RFFT.RCpair(Float64, sz, region)
        r = real(pair)
        b = rand(eltype(r), size(r))
        copy!(r, b)
        RFFT.rfft!(pair)
        RFFT.irfft!(pair)
        @test_approx_eq r b
        pfwd = RFFT.plan_rfft!(pair)
        pinv = RFFT.plan_irfft!(pair)
        pinv(pfwd(pair))
        @test_approx_eq r b
    end
end
