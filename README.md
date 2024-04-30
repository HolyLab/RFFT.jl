# RFFT.jl

In-place real FFTs for Julia. Supports "plans" to optimize the algorithm for transformations that you'll perform many times.

For example
```julia
import RFFT

a = rand(Float64, 100, 150)

# initialize a buffer 'RCpair' that contains a real and complex space
buf = RFFT.RCpair{Float64}(undef, size(a))
```

`real(buf)` views the underlying memory buffer as an array reals, while `complex(buf)` views the same
memory buffer as an array of complexes. The user is responsible for keeping track of which view is currently relevant.

If you'll be performing lots of FFTs on this buffer, it's best to create an optimized plan.

```julia
# create the plan
plan = RFFT.plan_rfft!(buf; flags=FFTW.MEASURE)

# use the plan and buffer on a new array
new = rand(Float64, 100, 150)
copy!(buf, new)
new_fft = plan(buf)

```

`RCpair` can be used to implement fast convolutions and many other fourier-based operations on real-valued data.
