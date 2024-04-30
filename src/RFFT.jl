"""
# RFFT

Highlights of the RFFT package:
- [`RCpair`](@ref): a buffer for in-place real-to-complex FFTs
- [`plan_rfft!`](@ref): create a plan for in-place real-to-complex FFTs
- [`plan_irfft!`](@ref): create a plan for in-place complex-to-real FFTs

Use `real(RC)` and `complex(RC)` to access the real and complex views of the buffer, respectively.
`copy!(RC, A)` can be used to copy data (either real- or complex-valued) into the buffer.
"""
module RFFT

using FFTW, LinearAlgebra

export RCpair, plan_rfft!, plan_irfft!, rfft!, irfft!, normalization

import Base: real, complex, copy, copy!

struct RCpair{T<:AbstractFloat,N,RType<:AbstractArray{T,N},CType<:AbstractArray{Complex{T},N}}
    R::RType
    C::CType
    dims::Vector{Int}
end

"""
    RCPair{T<:AbstractFloat}(undef, realsize::Dims, dims=1:length(realsize))

Create a buffer for performing in-place fourier transforms (and inverse) on real-valued data.
A single underlying buffer can be viewed either as real data or as complex data:

```julia
RC = RCpair{Float64}(undef, (10, 10))
real(RC) # 10×10 real array
complex(RC) # 6×10 complex array
```

`dims` can be used to control which dimensions are transformed.

The user is responsible to keep track of the current state of the buffer:

```julia
copy!(RC, rand(10, 10))   # copies real-valued data into the buffer; `real(RC)` is the relevant view
rfft!(RC)                 # computes the FFT of the real-valued data; now `complex(RC)` is the relevant view
irfft!(RC)                # computes the inverse FFT of the complex-valued data; now `real(RC)` is the relevant view
```
"""
function RCpair{T}(::UndefInitializer, realsize::Dims{N}, dims=1:length(realsize)) where {T<:AbstractFloat,N}
    sz = [realsize...]
    firstdim = dims[1]
    sz[firstdim] = realsize[firstdim]>>1 + 1
    sz2 = copy(sz)
    sz2[firstdim] *= 2
    R = Array{T,N}(undef, (sz2...,)::Dims{N})
    C = unsafe_wrap(Array, convert(Ptr{Complex{T}}, pointer(R)), (sz...,)::Dims{N}) # work around performance problems of reinterpretarray
    RCpair(view(R, map(n->1:n, realsize)...), C, [dims...])
end

RCpair(A::Array{T}, dims=1:ndims(A)) where {T<:AbstractFloat} = copy!(RCpair{T}(undef, size(A), dims), A)

real(RC::RCpair)    = RC.R
complex(RC::RCpair) = RC.C

copy!(RC::RCpair, A::AbstractArray{T}) where {T<:Real} = (copy!(RC.R, A); RC)
copy!(RC::RCpair, A::AbstractArray{T}) where {T<:Complex} = (copy!(RC.C, A); RC)
function copy(RC::RCpair{T,N}) where {T,N}
    C = copy(RC.C)
    R = reshape(reinterpret(T, C), size(parent(RC.R)))
    RCpair(view(R, RC.R.indices...), C, copy(RC.dims))
end

# New API
rplan_fwd(R, C, dims, flags, tlim) =
    FFTW.rFFTWPlan{eltype(R),FFTW.FORWARD,true,ndims(R)}(R, C, dims, flags, tlim)
rplan_inv(R, C, dims, flags, tlim) =
   FFTW.rFFTWPlan{eltype(R),FFTW.BACKWARD,true,ndims(R)}(R, C, dims, flags, tlim)

"""
    plan = plan_rfft!(RC::RCpair; flags=FFTW.ESTIMATE, timelimit=FFTW.NO_TIMELIMIT)

Create a plan for performing the real-to-complex FFT on the data in `RC`.
Perform the FFT with `plan(RC)`.

Planning allows you to optimize the performance of (I)FFTs for particular sizes of arrays.
See the FFTW documentation for more information about planning.
"""
function plan_rfft!(RC::RCpair{T}; flags::Integer = FFTW.ESTIMATE, timelimit::Real = FFTW.NO_TIMELIMIT) where T
    p = rplan_fwd(RC.R, RC.C, RC.dims, flags, timelimit)
    return Z::RCpair -> begin
        FFTW.assert_applicable(p, Z.R, Z.C)
        FFTW.unsafe_execute!(p, Z.R, Z.C)
        return Z
    end
end

"""
    plan = plan_rfft!(RC::RCpair; flags=FFTW.ESTIMATE, timelimit=FFTW.NO_TIMELIMIT)

Create a plan for performing the real-to-complex FFT on the data in `RC`.
Perform the FFT with `plan(RC)`.

Planning allows you to optimize the performance of (I)FFTs for particular sizes of arrays.
See the FFTW documentation for more information about planning.
"""
function plan_irfft!(RC::RCpair{T}; flags::Integer = FFTW.ESTIMATE, timelimit::Real = FFTW.NO_TIMELIMIT) where T
    p = rplan_inv(RC.C, RC.R, RC.dims, flags, timelimit)
    return Z::RCpair -> begin
        FFTW.assert_applicable(p, Z.C, Z.R)
        FFTW.unsafe_execute!(p, Z.C, Z.R)
        rmul!(Z.R, 1 / prod(size(Z.R)[Z.dims]))
        return Z
    end
end
function rfft!(RC::RCpair{T}) where T
    p = rplan_fwd(RC.R, RC.C, RC.dims, FFTW.ESTIMATE, FFTW.NO_TIMELIMIT)
    FFTW.unsafe_execute!(p, RC.R, RC.C)
    return RC
end
function irfft!(RC::RCpair{T}) where T
    p = rplan_inv(RC.C, RC.R, RC.dims, FFTW.ESTIMATE, FFTW.NO_TIMELIMIT)
    FFTW.unsafe_execute!(p, RC.C, RC.R)
    rmul!(RC.R, 1 / prod(size(RC.R)[RC.dims]))
    return RC
end

end
