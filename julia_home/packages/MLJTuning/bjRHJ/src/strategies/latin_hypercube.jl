"""
    LatinHypercube(gens = 1,
                   popsize = 100,
                   ntour = 2,
                   ptour = 0.8.,
                   interSampleWeight = 1.0,
                   ae_power = 2,
                   periodic_ae = false,
                   rng=Random.GLOBAL_RNG)

Instantiate grid-based hyperparameter tuning strategy using the
library [LatinHypercubeSampling.jl](https://github.com/MrUrq/LatinHypercubeSampling.jl).

An optimised Latin Hypercube sampling plan is created using a genetic
based optimization algorithm based on the inverse of the Audze-Eglais
function.  The optimization is run for `nGenerations` and creates `n`
models for evaluation, where `n` is specified by a corresponding `TunedModel`
instance, as in

    tuned_model = TunedModel(model=...,
                             tuning=LatinHypercube(...),
                             range=...,
                             measures=...,
                             n=...)

(See [`TunedModel`](@ref) for complete options.)

To use a periodic version of the Audze-Eglais function (to reduce
clustering along the boundaries) specify `periodic_ae = true`.

### Supported ranges:

A single one-dimensional range or vector of one-dimensioinal ranges
can be specified. Specifically, in `LatinHypercubeSampling` search,
the `range` field of a `TunedModel` instance can be:

- A single one-dimensional range - ie, `ParamRange` object - `r`, constructed
using the `range` method.

- Any vector of objects of the above form

Both `NumericRange`s and `NominalRange`s are supported, and
hyper-parameter values are sampled on a scale specified by the range
(eg, `r.scale = :log`).

"""
mutable struct LatinHypercube <: TuningStrategy
    gens::Int
    popsize::Int
    ntour::Int
    ptour::Number
    interSampleWeight::Number
    ae_power::Number
    periodic_ae::Bool
    rng::Random.AbstractRNG
end


function LatinHypercube(; gens = 1,
                        popsize = 100,
                        ntour = 2,
                        ptour = 0.8,
                        interSampleWeight = 1.0,
                        ae_power = 2,
                        periodic_ae = false,
                        rng=Random.GLOBAL_RNG)

    _rng = rng isa Integer ? Random.MersenneTwister(rng) : rng

    return LatinHypercube(gens, popsize, ntour,
                          ptour, interSampleWeight, ae_power,
                          periodic_ae, _rng)

end

function _create_bounds_and_dims_type(d,r)
    bounds = []
    dims_type = Array{LatinHypercubeSampling.LHCDimension}(undef,0)
    for i = 1:d
        if r[i] isa NumericRange
            if !(r[i].scale isa Symbol)
                throw(ArgumentError("Callable scale not supported."))
            end
            push!(dims_type,LatinHypercubeSampling.Continuous())
            if isfinite(r[i].lower) && isfinite(r[i].upper)
                push!(bounds,
                      Float64.([transform(MLJBase.Scale,
                                          MLJBase.scale(r[i].scale),
                                          r[i].lower),
                                transform(MLJBase.Scale,
                                          MLJBase.scale(r[i].scale),
                                          r[i].upper)]))
            elseif !isfinite(r[i].lower) && isfinite(r[i].upper)
                push!(bounds,
                      Float64.([transform(MLJBase.Scale,
                                          MLJBase.scale(r[i].scale),
                                          r[i].upper - 2*r[i].unit),
                                transform(MLJBase.Scale,
                                          MLJBase.scale(r[i].scale),
                                          r[i].upper)]))
            elseif isfinite(r[i].lower) && !isfinite(r[i].upper)
                push!(bounds, Float64.([transform(MLJBase.Scale,
                                                 MLJBase.scale(r[i].scale),
                                                 r[i].lower),
                                       transform(MLJBase.Scale,
                                                 MLJBase.scale(r[i].scale),
                                                 r[i].lower + 2*r[i].unit)]))
            else
                push!(bounds, Float64.([transform(MLJBase.Scale,
                                                 MLJBase.scale(r[i].scale),
                                                 r[i].origin - r[i].unit),
                                       transform(MLJBase.Scale,
                                                 MLJBase.scale(r[i].scale),
                                                 r[i].origin + r[i].unit)]))
            end
        else
            push!(dims_type,
                  LatinHypercubeSampling.Categorical(length(r[i].values),
                                                     1.0))
            push!(bounds,Float64.([1,length(r[i].values)]))
        end
    end
    return Tuple.(bounds), dims_type
end

# Hyper-param values generated by LatinHypercubeSampling library need
# further rescalings, and possibly rounding, depending on range type

# nominal ranges:
_transform(r::NominalRange, x) = r.values[round(Int, x)]

# numeric ranges:
_transform_noround(r, x) =
    inverse_transform(MLJBase.Scale, MLJBase.scale(r.scale), x)
_transform(r::NumericRange, x) = _transform_noround(r, x)
_transform(r::NumericRange{T}, x) where T<:Integer =
    round(T, _transform_noround(r, x))

function setup(tuning::LatinHypercube, model, range, n, verbosity)
    ranges = range isa AbstractVector ? range : [range, ]
    d = length(ranges)
    bounds, dims_type = _create_bounds_and_dims_type(d, ranges)
    plan, _ = LatinHypercubeSampling.LHCoptim(n, d, tuning.gens,
                    rng = tuning.rng,
                    popsize = tuning.popsize,
                    ntour = tuning.ntour,
                    ptour = tuning.ptour,
                    dims = dims_type,
                    interSampleWeight = tuning.interSampleWeight,
                    periodic_ae = tuning.periodic_ae,
                    ae_power = tuning.ae_power)
    scaled_plan = LatinHypercubeSampling.scaleLHC(plan, bounds)
    rescaled_plan = map(tuple(1:d...)) do k
        broadcast(x -> _transform(ranges[k], x), scaled_plan[:,k])
    end

    fields = map(r -> r.field, ranges)
    parameter_scales = scale.(ranges)
    models = makeLatinHypercube(model, fields, rescaled_plan)
    state = (models=models,
             fields=fields,
             parameter_scales=parameter_scales)
    return state
end

function MLJTuning.models(tuning::LatinHypercube,
                          model,
                          history,
                          state,
                          n_remaining,
                          verbosity)
     return state.models[_length(history) + 1:end], state
end

tuning_report(tuning::LatinHypercube, history, state) =
    (plotting = plotting_report(state.fields, state.parameter_scales, history),)

function makeLatinHypercube(prototype::Model, fields, plan)
    N = length(first(plan))
    map(1:N) do i
        clone = deepcopy(prototype)
        for k in eachindex(fields)
            recursive_setproperty!(clone,fields[k], plan[k][i])
        end
        clone
    end
end
