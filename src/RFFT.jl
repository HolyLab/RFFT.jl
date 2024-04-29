module RFFT

using FFTW, LinearAlgebra

export RCpair, plan_rfft!, plan_irfft!, rfft!, irfft!, normalization

import Base: real, complex, copy, copy!

mutable struct RCpair{T<:AbstractFloat,N,RType<:AbstractArray{T,N},CType<:AbstractArray{Complex{T},N}}
    R::RType
    C::CType
    dims::Vector{Int}
end

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
function plan_rfft!(RC::RCpair{T}; flags::Integer = FFTW.ESTIMATE, timelimit::Real = FFTW.NO_TIMELIMIT) where T
    p = rplan_fwd(RC.R, RC.C, RC.dims, flags, timelimit)
    return Z::RCpair -> begin
        FFTW.assert_applicable(p, Z.R, Z.C)
        FFTW.unsafe_execute!(p, Z.R, Z.C)
        return Z
    end
end
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
