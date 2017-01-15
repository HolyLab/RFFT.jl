__precompile__()

module RFFT

using Compat

export RCpair, plan_rfft!, plan_irfft!, rfft!, irfft!, normalization

import Base: real, complex, copy, copy!

type RCpair{T<:AbstractFloat,N}
    R::SubArray{T,N,Array{T,N},NTuple{N,UnitRange{Int}}}
    C::Array{Complex{T},N}
    region::Vector{Int}
end

function RCpair{T<:AbstractFloat}(realtype::Type{T}, realsize, region=1:length(realsize))
    sz = [realsize...]
    firstdim = region[1]
    sz[firstdim] = realsize[firstdim]>>1 + 1
    C = Array(Complex{T}, sz...)
    sz[firstdim] *= 2
    R = reinterpret(T, C, tuple(sz...))
    RCpair(Compat.view(R, map(n->1:n, realsize)...), C, [region...])
end
RCpair{T<:AbstractFloat}(A::Array{T}, region=1:ndims(A)) = copy!(RCpair(T, size(A), region), A)

real(RC::RCpair)    = RC.R
complex(RC::RCpair) = RC.C

copy!{T<:Real}(RC::RCpair, A::AbstractArray{T}) = (copy!(RC.R, A); RC)
function copy{T,N}(RC::RCpair{T,N})
    C = copy(RC.C)
    R = reinterpret(T, C, size(parent(RC.R)))
    RCpair(sub(R, RC.R.indexes), C, copy(RC.region))
end

if VERSION < v"0.4.0-dev+6068"
    # Old API
    function plan_rfft!{T}(RC::RCpair{T}; flags::Integer = FFTW.ESTIMATE, timelimit::Real = FFTW.NO_TIMELIMIT)
        p = FFTW.Plan(RC.R, RC.C, RC.region, flags, timelimit)
        return Z::RCpair{T} -> begin
            FFTW.assert_applicable(p, Z.R)
            FFTW.execute(p.plan, Z.R, Z.C)
            return Z
        end
    end

    function plan_irfft!{T}(RC::RCpair{T}; flags::Integer = FFTW.ESTIMATE, timelimit::Real = FFTW.NO_TIMELIMIT)
        p = FFTW.Plan(RC.C, RC.R, RC.region, flags, timelimit)
        return Z::RCpair{T} -> begin
            FFTW.assert_applicable(p, Z.C)
            FFTW.execute(p.plan, Z.C, Z.R)
            scale!(Z.R, 1 / prod(size(Z.R)[Z.region]))
            return Z
        end
    end

    function rfft!{T}(RC::RCpair{T})
        p = FFTW.Plan(RC.R, RC.C, RC.region, FFTW.ESTIMATE, FFTW.NO_TIMELIMIT)
        FFTW.execute(T, p.plan)
        return RC
    end

    function irfft!{T}(RC::RCpair{T})
        p = FFTW.Plan(RC.C, RC.R, RC.region, FFTW.ESTIMATE, FFTW.NO_TIMELIMIT)
        FFTW.execute(Complex{T}, p.plan)
        scale!(RC.R, 1 / prod(size(RC.R)[RC.region]))
        return RC
    end
else
    # New API
    rplan_fwd(R, C, region, flags, tlim) =
        FFTW.rFFTWPlan{eltype(R),FFTW.FORWARD,true,ndims(R)}(R, C, region, flags, tlim)
    rplan_inv(R, C, region, flags, tlim) =
       FFTW.rFFTWPlan{eltype(R),FFTW.BACKWARD,true,ndims(R)}(R, C, region, flags, tlim)

    function plan_rfft!{T}(RC::RCpair{T}; flags::Integer = FFTW.ESTIMATE, timelimit::Real = FFTW.NO_TIMELIMIT)
        p = rplan_fwd(RC.R, RC.C, RC.region, flags, timelimit)
        return Z::RCpair -> begin
            FFTW.assert_applicable(p, Z.R, Z.C)
            FFTW.unsafe_execute!(p, Z.R, Z.C)
            return Z
        end
    end

    function plan_irfft!{T}(RC::RCpair{T}; flags::Integer = FFTW.ESTIMATE, timelimit::Real = FFTW.NO_TIMELIMIT)
        p = rplan_inv(RC.C, RC.R, RC.region, flags, timelimit)
        return Z::RCpair -> begin
            FFTW.assert_applicable(p, Z.C, Z.R)
            FFTW.unsafe_execute!(p, Z.C, Z.R)
            scale!(Z.R, 1 / prod(size(Z.R)[Z.region]))
            return Z
        end
    end

    function rfft!{T}(RC::RCpair{T})
        p = rplan_fwd(RC.R, RC.C, RC.region, FFTW.ESTIMATE, FFTW.NO_TIMELIMIT)
        FFTW.unsafe_execute!(p, RC.R, RC.C)
        return RC
    end

    function irfft!{T}(RC::RCpair{T})
        p = rplan_inv(RC.C, RC.R, RC.region, FFTW.ESTIMATE, FFTW.NO_TIMELIMIT)
        FFTW.unsafe_execute!(p, RC.C, RC.R)
        scale!(RC.R, 1 / prod(size(RC.R)[RC.region]))
        return RC
    end
end

end
