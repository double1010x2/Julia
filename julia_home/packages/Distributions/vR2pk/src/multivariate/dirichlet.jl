"""
    Dirichlet

The [Dirichlet distribution](http://en.wikipedia.org/wiki/Dirichlet_distribution) is often
used as the conjugate prior for Categorical or Multinomial distributions.
The probability density function of a Dirichlet distribution with parameter
``\\alpha = (\\alpha_1, \\ldots, \\alpha_k)`` is:

```math
f(x; \\alpha) = \\frac{1}{B(\\alpha)} \\prod_{i=1}^k x_i^{\\alpha_i - 1}, \\quad \\text{ with }
B(\\alpha) = \\frac{\\prod_{i=1}^k \\Gamma(\\alpha_i)}{\\Gamma \\left( \\sum_{i=1}^k \\alpha_i \\right)},
\\quad x_1 + \\cdots + x_k = 1
```

```julia
# Let alpha be a vector
Dirichlet(alpha)         # Dirichlet distribution with parameter vector alpha

# Let a be a positive scalar
Dirichlet(k, a)          # Dirichlet distribution with parameter a * ones(k)
```
"""
struct Dirichlet{T<:Real,Ts<:AbstractVector{T},S<:Real} <: ContinuousMultivariateDistribution
    alpha::Ts
    alpha0::T
    lmnB::S

    function Dirichlet{T}(alpha::AbstractVector{T}; check_args=true) where T
        if check_args && !all(x -> x > zero(x), alpha)
            throw(ArgumentError("Dirichlet: alpha must be a positive vector."))
        end
        alpha0 = sum(alpha)
        lmnB = sum(loggamma, alpha) - loggamma(alpha0)
        new{T,typeof(alpha),typeof(lmnB)}(alpha, alpha0, lmnB)
    end
end

function Dirichlet(alpha::AbstractVector{<:Real}; check_args=true)
    Dirichlet{eltype(alpha)}(alpha; check_args=check_args)
end
Dirichlet(d::Integer, alpha::Real; kwargs...) = Dirichlet(Fill(alpha, d); kwargs...)

struct DirichletCanon{T<:Real,Ts<:AbstractVector{T}}
    alpha::Ts
end

length(d::DirichletCanon) = length(d.alpha)

Base.eltype(::Type{<:Dirichlet{T}}) where {T} = T

#### Conversions
convert(::Type{Dirichlet{T}}, cf::DirichletCanon) where {T<:Real} =
    Dirichlet(convert(AbstractVector{T}, cf.alpha))
convert(::Type{Dirichlet{T}}, alpha::AbstractVector{<:Real}) where {T<:Real} =
    Dirichlet(convert(AbstractVector{T}, alpha))
convert(::Type{Dirichlet{T}}, d::Dirichlet{<:Real}) where {T<:Real} =
    Dirichlet(convert(AbstractVector{T}, d.alpha))

convert(::Type{Dirichlet{T}}, cf::DirichletCanon{T}) where {T<:Real} = Dirichlet(cf.alpha)
convert(::Type{Dirichlet{T}}, alpha::AbstractVector{T}) where {T<:Real} =
    Dirichlet(alpha)
convert(::Type{Dirichlet{T}}, d::Dirichlet{T}) where {T<:Real} = d

Base.show(io::IO, d::Dirichlet) = show(io, d, (:alpha,))

# Properties

length(d::Dirichlet) = length(d.alpha)
mean(d::Dirichlet) = d.alpha .* inv(d.alpha0)
params(d::Dirichlet) = (d.alpha,)
@inline partype(d::Dirichlet{T}) where {T<:Real} = T

function var(d::Dirichlet)
    ??0 = d.alpha0
    c = inv(??0^2 * (??0 + 1))
    v = map(d.alpha) do ??i
        ??i * (??0 - ??i) * c
    end
    return v
end

function cov(d::Dirichlet)
    ?? = d.alpha
    ??0 = d.alpha0
    c = inv(??0^2 * (??0 + 1))

    T = typeof(zero(eltype(??))^2 * c)
    k = length(??)
    C = Matrix{T}(undef, k, k)
    for j = 1:k
        ??j = ??[j]
        ??jc = ??j * c
        for i in 1:(j-1)
            @inbounds C[i,j] = C[j,i]
        end
        @inbounds C[j,j] = (??0 - ??j) * ??jc
        for i in (j+1):k
            @inbounds C[i,j] = - ??[i] * ??jc
        end
    end

    return C
end

function entropy(d::Dirichlet)
    ??0 = d.alpha0
    ?? = d.alpha
    k = length(d.alpha)
    en = d.lmnB + (??0 - k) * digamma(??0) - sum(??j -> (??j - 1) * digamma(??j), ??)
    return en
end

function dirichlet_mode!(r::AbstractVector{<:Real}, ??::AbstractVector{<:Real}, ??0::Real)
    all(x -> x > 1, ??) || error("Dirichlet has a mode only when alpha[i] > 1 for all i")
    k = length(??)
    inv_s = inv(??0 - k)
    @. r = inv_s * (?? - 1)
    return r
end

function dirichlet_mode(??::AbstractVector{<:Real}, ??0::Real)
    all(x -> x > 1, ??) || error("Dirichlet has a mode only when alpha[i] > 1 for all i")
    inv_s = inv(??0 - length(??))
    r = map(??) do ??i
        inv_s * (??i - 1)
    end
    return r
end

mode(d::Dirichlet) = dirichlet_mode(d.alpha, d.alpha0)
mode(d::DirichletCanon) = dirichlet_mode(d.alpha, sum(d.alpha))

modes(d::Dirichlet) = [mode(d)]


# Evaluation

function insupport(d::Dirichlet, x::AbstractVector{<:Real})
    return length(d) == length(x) && !any(x -> x < zero(x), x) && sum(x) ??? 1
end

function _logpdf(d::Dirichlet, x::AbstractVector{<:Real})
    if !insupport(d, x)
        return xlogy(one(eltype(d.alpha)), zero(eltype(x))) - d.lmnB
    end
    a = d.alpha
    s = sum(xlogy(??i - 1, xi) for (??i, xi) in zip(d.alpha, x))
    return s - d.lmnB
end

# sampling

function _rand!(rng::AbstractRNG,
                d::Union{Dirichlet,DirichletCanon},
                x::AbstractVector{<:Real})
    for (i, ??i) in zip(eachindex(x), d.alpha)
        @inbounds x[i] = rand(rng, Gamma(??i))
    end
    lmul!(inv(sum(x)), x) # this returns x
end

function _rand!(rng::AbstractRNG,
                d::Dirichlet{T,<:FillArrays.AbstractFill{T}},
                x::AbstractVector{<:Real}) where {T<:Real}
    rand!(rng, Gamma(FillArrays.getindex_value(d.alpha)), x)
    lmul!(inv(sum(x)), x) # this returns x
end

#######################################
#
#  Estimation
#
#######################################

struct DirichletStats <: SufficientStats
    slogp::Vector{Float64}   # (weighted) sum of log(p)
    tw::Float64              # total sample weights

    DirichletStats(slogp::Vector{Float64}, tw::Real) = new(slogp, Float64(tw))
end

length(ss::DirichletStats) = length(s.slogp)

mean_logp(ss::DirichletStats) = ss.slogp * inv(ss.tw)

function suffstats(::Type{<:Dirichlet}, P::AbstractMatrix{Float64})
    K = size(P, 1)
    n = size(P, 2)
    slogp = zeros(K)
    for i = 1:n
        for k = 1:K
            @inbounds slogp[k] += log(P[k,i])
        end
    end
    DirichletStats(slogp, n)
end

function suffstats(::Type{<:Dirichlet}, P::AbstractMatrix{Float64},
                   w::AbstractArray{Float64})
    K = size(P, 1)
    n = size(P, 2)
    if length(w) != n
        throw(DimensionMismatch("Inconsistent argument dimensions."))
    end

    tw = 0.
    slogp = zeros(K)

    for i = 1:n
        @inbounds wi = w[i]
        tw += wi
        for k = 1:K
            @inbounds slogp[k] += log(P[k,i]) * wi
        end
    end
    DirichletStats(slogp, tw)
end

# fit_mle methods

## Initialization

function _dirichlet_mle_init2(??::Vector{Float64}, ??::Vector{Float64})
    K = length(??)

    ??0 = 0.
    for k = 1:K
        @inbounds ??k = ??[k]
        @inbounds ??k = ??[k]
        ak = (??k - ??k) / (??k - ??k * ??k)
        ??0 += ak
    end
    ??0 /= K

    lmul!(??0, ??)
end

function dirichlet_mle_init(P::AbstractMatrix{Float64})
    K = size(P, 1)
    n = size(P, 2)

    ?? = vec(sum(P, dims=2))       # E[p]
    ?? = vec(sum(abs2, P, dims=2)) # E[p^2]

    c = 1.0 / n
    ?? .*= c
    ?? .*= c

    _dirichlet_mle_init2(??, ??)
end

function dirichlet_mle_init(P::AbstractMatrix{Float64}, w::AbstractArray{Float64})
    K = size(P, 1)
    n = size(P, 2)

    ?? = zeros(K)  # E[p]
    ?? = zeros(K)  # E[p^2]
    tw = 0.0

    for i in 1:n
        @inbounds wi = w[i]
        tw += wi
        for k in 1:K
            pk = P[k, i]
            @inbounds ??[k] += pk * wi
            @inbounds ??[k] += pk * pk * wi
        end
    end

    c = 1.0 / tw
    ?? .*= c
    ?? .*= c

    _dirichlet_mle_init2(??, ??)
end

## Newton-Ralphson algorithm

function fit_dirichlet!(elogp::Vector{Float64}, ??::Vector{Float64};
    maxiter::Int=25, tol::Float64=1.0e-12, debug::Bool=false)
    # This function directly overrides ??

    K = length(elogp)
    length(??) == K || throw(DimensionMismatch("Inconsistent argument dimensions."))

    g = Vector{Float64}(undef, K)
    iq = Vector{Float64}(undef, K)
    ??0 = sum(??)

    if debug
        objv = dot(?? .- 1.0, elogp) + loggamma(??0) - sum(loggamma, ??)
    end

    t = 0
    converged = false
    while !converged && t < maxiter
        t += 1

        # compute gradient & Hessian
        # (b is computed as well)

        digam_??0 = digamma(??0)
        iz = 1.0 / trigamma(??0)
        gnorm = 0.
        b = 0.
        iqs = 0.

        for k = 1:K
            @inbounds ak = ??[k]
            @inbounds g[k] = gk = digam_??0 - digamma(ak) + elogp[k]
            @inbounds iq[k] = - 1.0 / trigamma(ak)

            @inbounds b += gk * iq[k]
            @inbounds iqs += iq[k]

            agk = abs(gk)
            if agk > gnorm
                gnorm = agk
            end
        end
        b /= (iz + iqs)

        # update ??

        for k = 1:K
            @inbounds ??[k] -= (g[k] - b) * iq[k]
            @inbounds if ??[k] < 1.0e-12
                ??[k] = 1.0e-12
            end
        end
        ??0 = sum(??)

        if debug
            prev_objv = objv
            objv = dot(?? .- 1.0, elogp) + loggamma(??0) - sum(loggamma, ??)
            @printf("Iter %4d: objv = %.4e  ch = %.3e  gnorm = %.3e\n",
                t, objv, objv - prev_objv, gnorm)
        end

        # determine convergence

        converged = gnorm < tol
    end

    if !converged
        throw(ErrorException("No convergence after $maxiter (maxiter) iterations."))
    end

    Dirichlet(??)
end


function fit_mle(::Type{T}, P::AbstractMatrix{Float64};
    init::Vector{Float64}=Float64[], maxiter::Int=25, tol::Float64=1.0e-12,
    debug::Bool=false) where {T<:Dirichlet}

    ?? = isempty(init) ? dirichlet_mle_init(P) : init
    elogp = mean_logp(suffstats(T, P))
    fit_dirichlet!(elogp, ??; maxiter=maxiter, tol=tol, debug=debug)
end

function fit_mle(::Type{<:Dirichlet}, P::AbstractMatrix{Float64},
                 w::AbstractArray{Float64};
    init::Vector{Float64}=Float64[], maxiter::Int=25, tol::Float64=1.0e-12,
    debug::Bool=false)

    n = size(P, 2)
    length(w) == n || throw(DimensionMismatch("Inconsistent argument dimensions."))

    ?? = isempty(init) ? dirichlet_mle_init(P, w) : init
    elogp = mean_logp(suffstats(Dirichlet, P, w))
    fit_dirichlet!(elogp, ??; maxiter=maxiter, tol=tol, debug=debug)
end
