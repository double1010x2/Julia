## CONSTANTS

const N_VALUES_THRESH = 16 # for BoxCoxTransformation


## DESCRIPTIONS (see also metadata at the bottom)

const FILL_IMPUTER_DESCR = "Imputes missing data with a fixed value "*
"computed on the non-missing values. A different imputing function "*
"can be specified for `Continuous`, `Count` and `Finite` data. "
const UNIVARIATE_FILL_IMPUTER_DESCR = "Univariate form of FillImpututer. "*
FILL_IMPUTER_DESCR
const FEATURE_SELECTOR_DESCR = "Filter features (columns) of a table by name."
const UNIVARIATE_STD_DESCR = "Standardize (whiten) univariate data."
const UNIVARIATE_DISCR_DESCR = "Discretize a continuous variable via "*
"quantiles."
const STANDARDIZER_DESCR = "Standardize (whiten) features (columns) "*
"of a table."
const UNIVARIATE_BOX_COX_DESCR = "Box-Cox transform univariate data."
const ONE_HOT_DESCR = "One-hot encode `Finite` (categorical) features "*
"(columns) of a table."
const CONTINUOUS_ENCODER_DESCR = "Convert all `Finite` (categorical) and "*
"`Count` features (columns) of a table to `Continuous` and drop all "*
" remaining non-`Continuous` features. "
"features. "
const UNIVARIATE_TIME_TYPE_TO_CONTINUOUS = "Transform univariate "*
"data with element scitype `ScientificDateTime` so that it has "*
"`Continuous` element scitype, according to a learned scale. "


##
## IMPUTER
##

round_median(v::AbstractVector) = v -> round(eltype(v), median(v))

_median(e)       = skipmissing(e) |> median
_round_median(e) = skipmissing(e) |> (f -> round(eltype(f), median(f)))
_mode(e)         = skipmissing(e) |> mode

@with_kw_noshow mutable struct UnivariateFillImputer <: Unsupervised
    continuous_fill::Function = _median
    count_fill::Function      = _round_median
    finite_fill::Function     = _mode
end

function MMI.fit(transformer::UnivariateFillImputer,
                      verbosity::Integer,
                      v)

    filler(v, ::Type) = throw(ArgumentError(
        "Imputation is not supported for vectors "*
        "of elscitype $(elscitype(v))."))
    filler(v, ::Type{<:Union{Continuous,Missing}}) =
        transformer.continuous_fill(v)
    filler(v, ::Type{<:Union{Count,Missing}}) =
        transformer.count_fill(v)
    filler(v, ::Type{<:Union{Finite,Missing}}) =
        transformer.finite_fill(v)

    fitresult = (filler=filler(v, elscitype(v)),)
    cache = nothing
    report = nothing

    return fitresult, cache, report

end

function replace_missing(::Type{<:Finite}, vnew, filler)
   all(in(levels(filler)), levels(vnew)) ||
        error(ArgumentError("The `column::AbstractVector{<:Finite}`"*
                            " to be transformed must contain the same levels"*
                            " as the categorical value to be imputed"))
   replace(vnew, missing => filler)

end

function replace_missing(::Type, vnew, filler)
   T = promote_type(nonmissing(eltype(vnew)), typeof(filler))
   w_tight = similar(vnew, T)
   @inbounds for i in eachindex(vnew)
        if ismissing(vnew[i])
           w_tight[i] = filler
        else
           w_tight[i] = vnew[i]
        end
   end
   return w_tight
end

function MMI.transform(transformer::UnivariateFillImputer,
                           fitresult,
                           vnew)

    filler = fitresult.filler

    scitype(filler) <: elscitype(vnew) ||
    error("Attempting to impute a value of scitype $(scitype(filler)) "*
    "into a vector of incompatible elscitype, namely $(elscitype(vnew)). ")

    if elscitype(vnew) >: Missing
        w_tight = replace_missing(nonmissing(elscitype(vnew)), vnew, filler)
    else
        w_tight = vnew
    end

    return w_tight
end

MMI.fitted_params(::UnivariateFillImputer, fitresult) = fitresult


"""
    FillImputer(
     features        = [],
     continuous_fill = e -> skipmissing(e) |> median
     count_fill      = e -> skipmissing(e) |> (f -> round(eltype(f), median(f)))
     finite_fill     = e -> skipmissing(e) |> mode

$FILL_IMPUTER_DESCR

## Fields

* `continuous_fill`: function to use on `Continuous` data, by default
  the median

* `count_fill`: function to use on `Count` data, by default the
  rounded median

* `finite_fill`: function to use on `Multiclass` and `OrderedFactor`
  data (including binary data), by default the mode

"""
@with_kw_noshow mutable struct FillImputer <: Unsupervised
    features::Vector{Symbol}  = Symbol[]
    continuous_fill::Function = _median
    count_fill::Function      = _round_median
    finite_fill::Function     = _mode
end

function MMI.fit(transformer::FillImputer, verbosity::Int, X)

    s = schema(X)
    features_seen = s.names |> collect # "seen" = "seen in fit"
    scitypes_seen = s.scitypes |> collect

    features = isempty(transformer.features) ? features_seen :
        transformer.features

    issubset(features, features_seen) || throw(ArgumentError(
    "Some features specified do not exist in the supplied table. "))

    # get corresponding scitypes:
    mask = map(features_seen) do ftr
        ftr in features
    end
    features = @view features_seen[mask] # `features` re-ordered
    scitypes = @view scitypes_seen[mask]
    features_and_scitypes = zip(features, scitypes) #|> collect

    # now keep those features that are imputable:
    function isimputable(ftr, T::Type)
        if verbosity > 0 && !isempty(transformer.features)
            @info "Feature $ftr will not be imputed "*
            "(imputation for $T not supported). "
        end
        return false
    end
    isimputable(ftr, ::Type{<:Union{Continuous,Missing}}) = true
    isimputable(ftr, ::Type{<:Union{Count,Missing}}) = true
    isimputable(ftr, ::Type{<:Union{Finite,Missing}}) = true

    mask = map(features_and_scitypes) do tup
        isimputable(tup...)
    end
    features_to_be_imputed = @view features[mask]

    univariate_transformer =
        UnivariateFillImputer(continuous_fill=transformer.continuous_fill,
                              count_fill=transformer.count_fill,
                              finite_fill=transformer.finite_fill)
    univariate_fitresult(ftr) = MMI.fit(univariate_transformer,
                                            verbosity - 1,
                                            selectcols(X, ftr))[1]

    fitresult_given_feature =
        Dict(ftr=> univariate_fitresult(ftr) for ftr in features_to_be_imputed)

    fitresult = (features_seen=features_seen,
                 univariate_transformer=univariate_transformer,
                 fitresult_given_feature=fitresult_given_feature)
    report    = nothing
    cache     = nothing

    return fitresult, cache, report
end

function MMI.transform(transformer::FillImputer, fitresult, X)

    features_seen = fitresult.features_seen # seen in fit
    univariate_transformer = fitresult.univariate_transformer
    fitresult_given_feature = fitresult.fitresult_given_feature

    all_features = Tables.schema(X).names

    # check that no new features have appeared:
    all(e -> e in features_seen, all_features) || throw(ArgumentError(
        "Attempting to transform table with "*
        "feature labels not seen in fit.\n"*
        "Features seen in fit = $features_seen.\n"*
        "Current features = $([all_features...]). "))

    features = keys(fitresult_given_feature)

    cols = map(all_features) do ftr
        col = MMI.selectcols(X, ftr)
        if ftr in features
            fr = fitresult_given_feature[ftr]
            return transform(univariate_transformer, fr, col)
        end
        return col
    end

    named_cols = NamedTuple{all_features}(tuple(cols...))
    return MMI.table(named_cols, prototype=X)

end

function MMI.fitted_params(::FillImputer, fr)
    dict = fr.fitresult_given_feature
    filler_given_feature = Dict(ftr=>dict[ftr].filler for ftr in keys(dict))
    return (features_seen_in_fit=fr.features_seen,
            univariate_transformer=fr.univariate_transformer,
            filler_given_feature=filler_given_feature)
end


##
## FOR FEATURE (COLUMN) SELECTION
##

"""
    FeatureSelector(features=Symbol[], ignore=false)

An unsupervised model for filtering features (columns) of a table.
Only those features encountered during fitting will appear in
transformed tables if `features` is empty (the default).
Alternatively, if a non-empty `features` is specified, then only the
specified features encountered during fitting are used (`ignore=false`) or all features
encountered during fitting which are not named in `features` are used (`ignore=true`).

Throws an error if a recorded or specified feature is not present in the transformation
input.

Instead of supplying a features vector, a Bool-valued callable with one argument
can be also be specified. For example, specifying `FeatureSelector(features =
name -> name in [:x1, :x3], ignore = true)` has the same effect as
`FeatureSelector(features = [:x1, :x3], ignore = true)`, namely to select
 all features, with the exception of `:x1` and `:x3`.

# Example

```
julia> X = (ordinal1 = [1, 2, 3],
            ordinal2 = coerce([:x, :y, :x], OrderedFactor),
            ordinal3 = [10.0, 20.0, 30.0],
            ordinal4 = [-20.0, -30.0, -40.0],
            nominal = coerce(["Your father", "he", "is"], Multiclass));

julia> select1 = FeatureSelector();

julia> transform(fit!(machine(select1, X)), X)
[ Info: Training Machine{FeatureSelector} @811.
(ordinal1 = [1, 2, 3],
 ordinal2 = CategoricalValue{Symbol,UInt32}[:x, :y, :x],
 ordinal3 = [-1.0, 0.0, 1.0],
 ordinal4 = [1.0, 0.0, -1.0],
 nominal = CategoricalVale{String,UInt32}["Your father", "he", "is"],)

julia> select2 = FeatureSelector(features=[:ordinal3, ], ignore=true);

julia> transform(fit!(machine(select2, X)), X)
[ Info: Training Machine{FeatureSelector} @721.
(ordinal1 = [1, 2, 3],
 ordinal2 = CategoricalValue{Symbol,UInt32}[:x, :y, :x],
 ordinal4 = [-20.0, -30.0, -40.0],
 nominal = CategoricalValue{String,UInt32}["Your father", "he", "is"],)

```
"""
mutable struct FeatureSelector <: Unsupervised
    # features to be selected; empty means all
    features::Union{Vector{Symbol}, Function}
    ignore::Bool # features to be ignored
end

# keyword constructor
function FeatureSelector(
    ;
    features::Union{AbstractVector{Symbol}, Function}=Symbol[],
    ignore::Bool=false
)
    transformer = FeatureSelector(features, ignore)
    message = MMI.clean!(transformer)
    isempty(message) || throw(ArgumentError(message))
    return transformer
end

function MMI.clean!(transformer::FeatureSelector)
    err = ""
    if (
        typeof(transformer.features) <: AbstractVector{Symbol} &&
        isempty(transformer.features) &&
        transformer.ignore
    )
        err *= "Features to be ignored must be specified in features field."
    end
    return err
end

function MMI.fit(transformer::FeatureSelector, verbosity::Int, X)
    all_features = Tables.schema(X).names

    if transformer.features isa AbstractVector{Symbol}
        if isempty(transformer.features)
           features = collect(all_features)
        else
            features = if transformer.ignore
                !issubset(transformer.features, all_features) && verbosity > -1 &&
                @warn("Excluding non-existent feature(s).")
                filter!(all_features |> collect) do ftr
                   !(ftr in transformer.features)
                end
            else
                issubset(transformer.features, all_features) ||
                throw(ArgumentError("Attempting to select non-existent feature(s)."))
                transformer.features |> collect
            end
        end
    else
        features = if transformer.ignore
            filter!(all_features |> collect) do ftr
                !(transformer.features(ftr))
            end
        else
            filter!(all_features |> collect) do ftr
                transformer.features(ftr)
            end
        end
        isempty(features) && throw(
            ArgumentError("No feature(s) selected.\n The specified Bool-valued"*
              " callable with the `ignore` option set to `$(transformer.ignore)` "*
              "resulted in an empty feature set for selection")
         )
    end

    fitresult = features
    report = NamedTuple()
    return fitresult, nothing, report
end

MMI.fitted_params(::FeatureSelector, fitresult) = (features_to_keep=fitresult,)

function MMI.transform(::FeatureSelector, features, X)
    all(e -> e in Tables.schema(X).names, features) ||
        throw(ArgumentError("Supplied frame does not admit previously selected features."))
    return MMI.selectcols(X, features)
end

##
## UNIVARIATE Discretizer
##

# helper function:
reftype(::CategoricalArray{<:Any,<:Any,R}) where R = R

"""
    UnivariateDiscretizer(n_classes=512)

Returns an `MLJModel` for for discretizing any continuous vector `v`
 (`scitype(v) <: AbstractVector{Continuous}`), where `n_classes`
 describes the resolution of the discretization.

Transformed output `w` is a vector of ordered factors (`scitype(w) <:
 AbstractVector{<:OrderedFactor}`). Specifically, `w` is a
 `CategoricalVector`, with element type
 `CategoricalValue{R,R}`, where `R<Unsigned` is optimized.

The transformation is chosen so that the vector on which the
 transformer is fit has, in transformed form, an approximately uniform
 distribution of values.

### Example

    using MLJ
    t = UnivariateDiscretizer(n_classes=10)
    discretizer = machine(t, randn(1000))
    fit!(discretizer)
    v = rand(10)
    w = transform(discretizer, v)
    v_approx = inverse_transform(discretizer, w) # reconstruction of v from w

"""
@with_kw_noshow mutable struct UnivariateDiscretizer <:Unsupervised
    n_classes::Int = 512
end

struct UnivariateDiscretizerResult{C}
    odd_quantiles::Vector{Float64}
    even_quantiles::Vector{Float64}
    element::C
end

function MMI.fit(transformer::UnivariateDiscretizer, verbosity::Int, X)
    n_classes = transformer.n_classes
    quantiles = quantile(X, Array(range(0, stop=1, length=2*n_classes+1)))
    clipped_quantiles = quantiles[2:2*n_classes] # drop 0% and 100% quantiles

    # odd_quantiles for transforming, even_quantiles used for
    # inverse_transforming:
    odd_quantiles = clipped_quantiles[2:2:(2*n_classes-2)]
    even_quantiles = clipped_quantiles[1:2:(2*n_classes-1)]

    # determine optimal reference type for encoding as categorical:
    R = reftype(categorical(1:n_classes, compress=true))
    output_prototype = categorical(R(1):R(n_classes), compress=true, ordered=true)
    element = output_prototype[1]

    cache  = nothing
    report = NamedTuple()

    res = UnivariateDiscretizerResult(odd_quantiles, even_quantiles, element)
    return res, cache, report
end

# acts on scalars:
function transform_to_int(
            result::UnivariateDiscretizerResult{<:CategoricalValue{R}},
            r::Real) where R
    k = oneR = R(1)
    @inbounds for q in result.odd_quantiles
        if r > q
            k += oneR
        end
    end
    return k
end

# transforming scalars:
MMI.transform(::UnivariateDiscretizer, result, r::Real) =
    transform(result.element, transform_to_int(result, r))

# transforming vectors:
function MMI.transform(::UnivariateDiscretizer, result, v)
   w = [transform_to_int(result, r) for r in v]
   return transform(result.element, w)
end

# inverse_transforming raw scalars:
function MMI.inverse_transform(
    transformer::UnivariateDiscretizer, result , k::Integer)
    k <= transformer.n_classes && k > 0 ||
        error("Cannot transform an integer outside the range "*
              "`[1, n_classes]`, where `n_classes = $(transformer.n_classes)`")
    return result.even_quantiles[k]
end

# inverse transforming a categorical value:
function MMI.inverse_transform(
    transformer::UnivariateDiscretizer, result, e::CategoricalValue)
    k = CategoricalArrays.DataAPI.unwrap(e)
    return inverse_transform(transformer, result, k)
end

# inverse transforming raw vectors:
MMI.inverse_transform(transformer::UnivariateDiscretizer, result,
                          w::AbstractVector{<:Integer}) =
      [inverse_transform(transformer, result, k) for k in w]

# inverse transforming vectors of categorical elements:
function MMI.inverse_transform(transformer::UnivariateDiscretizer, result,
                          wcat::AbstractVector{<:CategoricalValue})
    w = MMI.int(wcat)
    return [inverse_transform(transformer, result, k) for k in w]
end


## UNIVARIATE STANDARDIZATION

"""
    UnivariateStandardizer()

Unsupervised model for standardizing (whitening) univariate data.
"""
mutable struct UnivariateStandardizer <: Unsupervised end

function MMI.fit(transformer::UnivariateStandardizer, verbosity::Int,
             v::AbstractVector{T}) where T<:Real
    std(v) > eps(Float64) ||
        @warn "Extremely small standard deviation encountered in standardization."
    fitresult = (mean(v), std(v))
    cache = nothing
    report = NamedTuple()
    return fitresult, cache, report
end


MMI.fitted_params(::UnivariateStandardizer, fitresult) =
    (mean_and_std = fitresult, )


# for transforming single value:
function MMI.transform(transformer::UnivariateStandardizer, fitresult, x::Real)
    mu, sigma = fitresult
    return (x - mu)/sigma
end

# for transforming vector:
MMI.transform(transformer::UnivariateStandardizer, fitresult, v) =
              [transform(transformer, fitresult, x) for x in v]

# for single values:
function MMI.inverse_transform(transformer::UnivariateStandardizer, fitresult, y::Real)
    mu, sigma = fitresult
    return mu + y*sigma
end

# for vectors:
MMI.inverse_transform(transformer::UnivariateStandardizer, fitresult, w) =
    [inverse_transform(transformer, fitresult, y) for y in w]


## CONTINUOUS TRANSFORM OF TIME TYPE FEATURES

"""
    UnivariateTimeTypeToContinuous(zero_time=nothing, step=Hour(24))

Convert a `Date`, `DateTime`, and `Time` vector to `Float64` by
assuming `0.0` corresponds to the `zero_time` parameter and the time
increment to reach `1.0` is given by the `step` parameter. The type of
`zero_time` should match the type of the column if provided. If not
provided, then `zero_time` is inferred as the minimum time found in
the data when `fit` is called.

"""
mutable struct UnivariateTimeTypeToContinuous <: Unsupervised
    zero_time::Union{Nothing, TimeType}
    step::Period
end

function UnivariateTimeTypeToContinuous(;
    zero_time=nothing, step=Dates.Hour(24))
    model = UnivariateTimeTypeToContinuous(zero_time, step)
    message = MMI.clean!(model)
    isempty(message) || @warn message
    return model
end

function MMI.clean!(model::UnivariateTimeTypeToContinuous)
    # Step must be able to be added to zero_time if provided.
    msg = ""
    if model.zero_time !== nothing
        try
            tmp = model.zero_time + model.step
        catch err
            if err isa MethodError
                model.zero_time, model.step, status, msg = _fix_zero_time_step(
                    model.zero_time, model.step)
                if status === :error
                    # Unable to resolve, rethrow original error.
                    throw(err)
                end
            else
                throw(err)
            end
        end
    end
    return msg
end

function _fix_zero_time_step(zero_time, step)
    # Cannot add time parts to dates nor date parts to times.
    # If a mismatch is encountered. Conversion from date parts to time parts
    # is possible, but not from time parts to date parts because we cannot
    # represent fractional date parts.
    msg = ""
    if zero_time isa Dates.Date && step isa Dates.TimePeriod
        # Convert zero_time to a DateTime to resolve conflict.
        if step % Hour(24) === Hour(0)
            # We can convert step to Day safely
            msg = "Cannot add `TimePeriod` `step` to `Date` `zero_time`. Converting `step` to `Day`."
            step = convert(Day, step)
        else
            # We need datetime to be compatible with the step.
            msg = "Cannot add `TimePeriod` `step` to `Date` `zero_time`. Converting `zero_time` to `DateTime`."
            zero_time = convert(DateTime, zero_time)
        end
        return zero_time, step, :success, msg
    elseif zero_time isa Dates.Time && step isa Dates.DatePeriod
        # Convert step to Hour if possible. This will fail for
        # isa(step, Month)
        msg = "Cannot add `DatePeriod` `step` to `Time` `zero_time`. Converting `step` to `Hour`."
        step = convert(Hour, step)
        return zero_time, step, :success, msg
    else
        return zero_time, step, :error, msg
    end
end

function MMI.fit(model::UnivariateTimeTypeToContinuous, verbosity::Int, X)
    if model.zero_time !== nothing
        min_dt = model.zero_time
        step = model.step
        # Check zero_time is compatible with X
        example = first(X)
        try
            X - min_dt
        catch err
            if err isa MethodError
                @warn "`$(typeof(min_dt))` `zero_time` is not compatible with `$(eltype(X))` vector. Attempting to convert `zero_time`."
                min_dt = convert(eltype(X), min_dt)
            else
                throw(err)
            end
        end
    else
        min_dt = minimum(X)
        step = model.step
        message = ""
        try
            min_dt + step
        catch err
            if err isa MethodError
                min_dt, step, status, message = _fix_zero_time_step(min_dt, step)
                if status === :error
                    # Unable to resolve, rethrow original error.
                    throw(err)
                end
            else
                throw(err)
            end
        end
        isempty(message) || @warn message
    end
    cache = nothing
    report = nothing
    fitresult = (min_dt, step)
    return fitresult, cache, report
end

function MMI.transform(model::UnivariateTimeTypeToContinuous, fitresult, X)
    min_dt, step = fitresult
    if typeof(min_dt) ??? eltype(X)
        # Cannot run if eltype in transform differs from zero_time from fit.
        throw(ArgumentError("Different `TimeType` encountered during `transform` than expected from `fit`. Found `$(eltype(X))`, expected `$(typeof(min_dt))`"))
    end
    # Set the size of a single step.
    next_time = min_dt + step
    if next_time == min_dt
        # Time type loops if step is a multiple of Hour(24), so calculate the
        # number of multiples, then re-scale to Hour(12) and adjust delta to match original.
        m = step / Dates.Hour(12)
        delta = m * (
            Float64(Dates.value(min_dt + Dates.Hour(12)) - Dates.value(min_dt)))
    else
        delta = Float64(Dates.value(min_dt + step) - Dates.value(min_dt))
    end
    return @. Float64(Dates.value(X - min_dt)) / delta
end


## STANDARDIZATION OF ORDINAL FEATURES OF TABULAR DATA

"""
    Standardizer(; features=Symbol[],
                   ignore=false,
                   ordered_factor=false,
                   count=false)

Unsupervised model for standardizing (whitening) the columns of
tabular data.  If `features` is unspecified then all columns
having `Continuous` element scitype are standardized. Otherwise, the
features standardized are the `Continuous` features named in
`features` (`ignore=false`) or `Continuous` features not named in
`features` (`ignore=true`). To allow standarization of `Count` or
`OrderedFactor` features as well, set the appropriate flag to true.

Instead of supplying a features vector, a Bool-valued callable with one
argument can be also be specified. For example, specifying
`Standardizer(features = name -> name in [:x1, :x3], ignore = true, count=true)`
has the same effect as `Standardizer(features = [:x1, :x3], ignore = true,
count=true)`, namely to standardise all `Continuous` and `Count` features,
with the exception of `:x1` and `:x3`.

The `inverse_tranform` method is supported provided `count=false` and
`ordered_factor=false` at time of fit.

# Example

```
X = (ordinal1 = [1, 2, 3],
     ordinal2 = coerce([:x, :y, :x], OrderedFactor),
     ordinal3 = [10.0, 20.0, 30.0],
     ordinal4 = [-20.0, -30.0, -40.0],
     nominal = coerce(["Your father", "he", "is"], Multiclass));
stand1 = Standardizer();
julia> transform(fit!(machine(stand1, X)), X)
[ Info: Training Machine{Standardizer} @ 7???97.
(ordinal1 = [1, 2, 3],
 ordinal2 = CategoricalValue{Symbol,UInt32}[:x, :y, :x],
 ordinal3 = [-1.0, 0.0, 1.0],
 ordinal4 = [1.0, 0.0, -1.0],
 nominal = CategoricalVale{String,UInt32}["Your father", "he", "is"],)

stand2 = Standardizer(features=[:ordinal3, ], ignore=true, count=true);
julia> transform(fit!(machine(stand2, X)), X)
[ Info: Training Machine{Standardizer} @ 1???87.
(ordinal1 = [-1.0, 0.0, 1.0],
 ordinal2 = CategoricalValue{Symbol,UInt32}[:x, :y, :x],
 ordinal3 = [10.0, 20.0, 30.0],
 ordinal4 = [1.0, 0.0, -1.0],
 nominal = CategoricalValue{String,UInt32}["Your father", "he", "is"],)
```

"""
mutable struct Standardizer <: Unsupervised
    # features to be standardized; empty means all
    features::Union{AbstractVector{Symbol}, Function}
    ignore::Bool # features to be ignored
    ordered_factor::Bool
    count::Bool
end

# keyword constructor
function Standardizer(
    ;
    features::Union{AbstractVector{Symbol}, Function}=Symbol[],
    ignore::Bool=false,
    ordered_factor::Bool=false,
    count::Bool=false
)
    transformer = Standardizer(features, ignore, ordered_factor, count)
    message = MMI.clean!(transformer)
    isempty(message) || throw(ArgumentError(message))
    return transformer
end

function MMI.clean!(transformer::Standardizer)
    err = ""
    if (
        typeof(transformer.features) <: AbstractVector{Symbol} &&
        isempty(transformer.features) &&
        transformer.ignore
    )
        err *= "Features to be ignored must be specified in features field."
    end
    return err
end

function MMI.fit(transformer::Standardizer, verbosity::Int, X)

    # if not a table, it must be an abstract vector, eltpye AbstractFloat:
    is_univariate = !Tables.istable(X)

    # are we attempting to standardize Count or OrderedFactor?
    is_invertible = !transformer.count && !transformer.ordered_factor

    # initialize fitresult:
    fitresult_given_feature = Dict{Symbol,Tuple{Float64,Float64}}()

    # special univariate case:
    if is_univariate
        fitresult_given_feature[:unnamed] =
            MMI.fit(UnivariateStandardizer(), verbosity - 1, X)[1]
        return (is_univariate=true,
                is_invertible=true,
                fitresult_given_feature=fitresult_given_feature),
        nothing, nothing
    end

    all_features = Tables.schema(X).names
    feature_scitypes =
        collect(elscitype(selectcols(X, c)) for c in all_features)
    scitypes = Vector{Type}([Continuous])
    transformer.ordered_factor && push!(scitypes, OrderedFactor)
    transformer.count && push!(scitypes, Count)
    AllowedScitype = Union{scitypes...}

    # determine indices of all_features to be transformed
    if transformer.features isa AbstractVector{Symbol}
        if isempty(transformer.features)
            cols_to_fit = filter!(eachindex(all_features) |> collect) do j
                feature_scitypes[j] <: AllowedScitype
            end
        else
            !issubset(transformer.features, all_features) && verbosity > -1 &&
                @warn "Some specified features not present in table to be fit. "
            cols_to_fit = filter!(eachindex(all_features) |> collect) do j
                ifelse(
                    transformer.ignore,
                    !(all_features[j] in transformer.features) &&
                        feature_scitypes[j] <: AllowedScitype,
                    (all_features[j] in transformer.features) &&
                        feature_scitypes[j] <: AllowedScitype
                )
            end
        end
    else
        cols_to_fit = filter!(eachindex(all_features) |> collect) do j
            ifelse(
                transformer.ignore,
                !(transformer.features(all_features[j])) &&
                    feature_scitypes[j] <: AllowedScitype,
                (transformer.features(all_features[j])) &&
                    feature_scitypes[j] <: AllowedScitype
            )
        end
    end
    fitresult_given_feature = Dict{Symbol,Tuple{Float64,Float64}}()

    isempty(cols_to_fit) && verbosity > -1 &&
        @warn "No features to standarize."

    # fit each feature and add result to above dict
    verbosity < 2 || @info "Features standarized: "
    for j in cols_to_fit
        col_data = if (feature_scitypes[j] <: OrderedFactor)
            coerce(selectcols(X, j), Continuous)
        else
            selectcols(X, j)
        end
        col_fitresult, cache, report =
            MMI.fit(UnivariateStandardizer(), verbosity - 1, col_data)
        fitresult_given_feature[all_features[j]] = col_fitresult
        verbosity < 2 ||
            @info "  :$(all_features[j])    mu=$(col_fitresult[1])  sigma=$(col_fitresult[2])"
    end

    fitresult = (is_univariate=false, is_invertible=is_invertible,
                 fitresult_given_feature=fitresult_given_feature)
    cache = nothing
    report = (features_fit=keys(fitresult_given_feature),)

    return fitresult, cache, report
end

function MMI.fitted_params(::Standardizer, fitresult)
    is_univariate, _, dic = fitresult
    is_univariate &&
        return fitted_params(UnivariateStandardizer(), dic[:unnamed])
    return (mean_and_std_given_feature=dic)
end

MMI.transform(::Standardizer, fitresult, X) =
    _standardize(transform, fitresult, X)

function MMI.inverse_transform(::Standardizer, fitresult, X)
    fitresult.is_invertible ||
        error("Inverse standardization is not supported when `count=true` "*
              "or `ordered_factor=true` during fit. ")
    return _standardize(inverse_transform, fitresult, X)
end

function _standardize(operation, fitresult, X)

    # `fitresult` is dict of column fitresults, keyed on feature names
    is_univariate, _, fitresult_given_feature = fitresult

    if is_univariate
        univariate_fitresult = fitresult_given_feature[:unnamed]
        return operation(UnivariateStandardizer(), univariate_fitresult, X)
    end

    features_to_be_transformed = keys(fitresult_given_feature)

    all_features = Tables.schema(X).names

    all(e -> e in all_features, features_to_be_transformed) ||
        error("Attempting to transform data with incompatible feature labels.")

    col_transformer = UnivariateStandardizer()

    cols = map(all_features) do ftr
        ftr_data = selectcols(X, ftr)
        if ftr in features_to_be_transformed
            col_to_transform = coerce(ftr_data, Continuous)
            operation(col_transformer,
                      fitresult_given_feature[ftr],
                      col_to_transform)
        else
            ftr_data
        end
    end

    named_cols = NamedTuple{all_features}(tuple(cols...))

    return MMI.table(named_cols, prototype=X)
end



##
## UNIVARIATE BOX-COX TRANSFORMATIONS
##

function standardize(v)
    map(v) do x
        (x - mean(v))/std(v)
    end
end

function midpoints(v::AbstractVector{T}) where T <: Real
    return [0.5*(v[i] + v[i + 1]) for i in 1:(length(v) -1)]
end

function normality(v)
    n  = length(v)
    v  = standardize(convert(Vector{Float64}, v))
    # sort and replace with midpoints
    v = midpoints(sort!(v))
    # find the (approximate) expected value of the size (n-1)-ordered statistics for
    # standard normal:
    d = Distributions.Normal(0,1)
    w = map(collect(1:(n-1))/n) do x
        quantile(d, x)
    end
    return cor(v, w)
end

function boxcox(lambda, c, x::Real)
    c + x >= 0 || throw(DomainError)
    if lambda == 0.0
        c + x > 0 || throw(DomainError)
        return log(c + x)
    end
    return ((c + x)^lambda - 1)/lambda
end

boxcox(lambda, c, v::AbstractVector{T}) where T <: Real =
    [boxcox(lambda, c, x) for x in v]


"""
    UnivariateBoxCoxTransformer(; n=171, shift=false)

Unsupervised model specifying a univariate Box-Cox
transformation of a single variable taking non-negative values, with a
possible preliminary shift. Such a transformation is of the form

    x -> ((x + c)^?? - 1)/?? for ?? not 0
    x -> log(x + c) for ?? = 0

On fitting to data `n` different values of the Box-Cox
exponent ?? (between `-0.4` and `3`) are searched to fix the value
maximizing normality. If `shift=true` and zero values are encountered
in the data then the transformation sought includes a preliminary
positive shift `c` of `0.2` times the data mean. If there are no zero
values, then no shift is applied.

"""
@with_kw_noshow mutable struct UnivariateBoxCoxTransformer <: Unsupervised
    n::Int      = 171   # nbr values tried in optimizing exponent lambda
    shift::Bool = false # whether to shift data away from zero
end

function MMI.fit(transformer::UnivariateBoxCoxTransformer, verbosity::Int,
             v::AbstractVector{T}) where T <: Real

    m = minimum(v)
    m >= 0 || error("Cannot perform a Box-Cox transformation on negative data.")

    c = 0.0 # default
    if transformer.shift
        if m == 0
            c = 0.2*mean(v)
        end
    else
        m != 0 || error("Zero value encountered in data being Box-Cox transformed.\n"*
                        "Consider calling `fit!` with `shift=true`.")
    end

    lambdas = range(-0.4, stop=3, length=transformer.n)
    scores = Float64[normality(boxcox(l, c, v)) for l in lambdas]
    lambda = lambdas[argmax(scores)]

    return  (lambda, c), nothing, NamedTuple()
end

MMI.fitted_params(::UnivariateBoxCoxTransformer, fitresult) =
    (??=fitresult[1], c=fitresult[2])

# for X scalar or vector:
MMI.transform(transformer::UnivariateBoxCoxTransformer, fitresult, X) =
    boxcox(fitresult..., X)

# scalar case:
function MMI.inverse_transform(transformer::UnivariateBoxCoxTransformer,
                           fitresult, x::Real)
    lambda, c = fitresult
    if lambda == 0
        return exp(x) - c
    else
        return (lambda*x + 1)^(1/lambda) - c
    end
end

# vector case:
function MMI.inverse_transform(transformer::UnivariateBoxCoxTransformer,
                           fitresult, w::AbstractVector{T}) where T <: Real
    return [inverse_transform(transformer, fitresult, y) for y in w]
end


## ONE HOT ENCODING

"""
    OneHotEncoder(; features=Symbol[],
                    ignore=false,
                    ordered_factor=true,
                    drop_last=false)

Unsupervised model for one-hot encoding the `Finite` features
(columns) of some table. If `features` is unspecified all features
with `Finite` element scitype are encoded. Otherwise, encoding is
applied to all `Finite` features named in `features` (`ignore=false`)
or all `Finite` features not named in features (`ignore=true`).

If `ordered_factor=false` then the above holds with `Finite` replaced
with `Multiclass`, ie `OrderedFactor` features are not transformed.

Specify `drop_last=true` if the column for the last level of each
categorical feature is to be dropped.

New data to be transformed may lack features present in the fit data,
but no *new* features can be present.

*Warning:* This transformer assumes that `levels(col)` for any
`Multiclass` or `OrderedFactor` column is the same in new data being
transformed as it is in the data used to fit the transformer.

### Example

```julia
X = (name=categorical(["Danesh", "Lee", "Mary", "John"]),
     grade=categorical([:A, :B, :A, :C], ordered=true),
     height=[1.85, 1.67, 1.5, 1.67],
     n_devices=[3, 2, 4, 3])
schema(X)

??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
??? _.names   ??? _.types                         ??? _.scitypes       ???
??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
??? name      ??? CategoricalValue{String,UInt32} ??? Multiclass{4}    ???
??? grade     ??? CategoricalValue{Symbol,UInt32} ??? OrderedFactor{3} ???
??? height    ??? Float64                         ??? Continuous       ???
??? n_devices ??? Int64                           ??? Count            ???
??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
_.nrows = 4

hot = OneHotEncoder(ordered_factor=true);
mach = fit!(machine(hot, X))
transform(mach, X) |> schema

?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
??? _.names      ??? _.types ??? _.scitypes ???
?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
??? name__Danesh ??? Float64 ??? Continuous ???
??? name__John   ??? Float64 ??? Continuous ???
??? name__Lee    ??? Float64 ??? Continuous ???
??? name__Mary   ??? Float64 ??? Continuous ???
??? grade__A     ??? Float64 ??? Continuous ???
??? grade__B     ??? Float64 ??? Continuous ???
??? grade__C     ??? Float64 ??? Continuous ???
??? height       ??? Float64 ??? Continuous ???
??? n_devices    ??? Int64   ??? Count      ???
?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
_.nrows = 4
```

"""
@with_kw_noshow mutable struct OneHotEncoder <: Unsupervised
    features::Vector{Symbol}   = Symbol[]
    drop_last::Bool            = false
    ordered_factor::Bool       = true
    ignore::Bool               = false
end

# we store the categorical refs for each feature to be encoded and the
# corresponing feature labels generated (called
# "names"). `all_features` is stored to ensure no new features appear
# in new input data, causing potential name clashes.
struct OneHotEncoderResult <: MMI.MLJType
    all_features::Vector{Symbol} # all feature labels
    ref_name_pairs_given_feature::Dict{Symbol,Vector{Pair{<:Unsigned,Symbol}}}
    fitted_levels_given_feature::Dict{Symbol, CategoricalArray}
end

# join feature and level into new label without clashing with anything
# in all_features:
function compound_label(all_features, feature, level)
    label = Symbol(string(feature, "__", level))
    # in the (rare) case subft is not a new feature label:
    while label in all_features
        label = Symbol(string(label,"_"))
    end
    return label
end

function MMI.fit(transformer::OneHotEncoder, verbosity::Int, X)

    all_features = Tables.schema(X).names # a tuple not vector

    if isempty(transformer.features)
        specified_features = collect(all_features)
    else
        if transformer.ignore
            specified_features = filter(all_features |> collect) do ftr
                !(ftr in transformer.features)
            end
        else
            specified_features = transformer.features
        end
    end

    ref_name_pairs_given_feature =
        Dict{Symbol,Vector{Pair{<:Unsigned,Symbol}}}()

    allowed_scitypes = ifelse(transformer.ordered_factor, Finite, Multiclass)
    fitted_levels_given_feature = Dict{Symbol, CategoricalArray}()
    col_scitypes = schema(X).scitypes
    # apply on each feature
    for j in eachindex(all_features)
        ftr = all_features[j]
        col = MMI.selectcols(X,j)
        T = col_scitypes[j]
        if T <: allowed_scitypes && ftr in specified_features
            ref_name_pairs_given_feature[ftr] = Pair{<:Unsigned,Symbol}[]
            shift = transformer.drop_last ? 1 : 0
            levels = MMI.classes(first(col))
            fitted_levels_given_feature[ftr] = levels
            if verbosity > 0
                @info "Spawning $(length(levels)-shift) sub-features "*
                "to one-hot encode feature :$ftr."
            end
            for level in levels[1:end-shift]
                ref = MMI.int(level)
                name = compound_label(all_features, ftr, level)
                push!(ref_name_pairs_given_feature[ftr], ref => name)
            end
        end
    end

    fitresult = OneHotEncoderResult(collect(all_features),
                                    ref_name_pairs_given_feature,
                                    fitted_levels_given_feature)

    # get new feature names
    d = ref_name_pairs_given_feature
    new_features = Symbol[]
    features_to_be_transformed = keys(d)
    for ftr in all_features
        if ftr in features_to_be_transformed
            append!(new_features, last.(d[ftr]))
        else
            push!(new_features, ftr)
        end
    end

    report = (features_to_be_encoded=
              collect(keys(ref_name_pairs_given_feature)),
              new_features=new_features)
    cache = nothing

    return fitresult, cache, report
end

# If v=categorical('a', 'a', 'b', 'a', 'c') and MMI.int(v[1]) = ref
# then `_hot(v, ref) = [true, true, false, true, false]`
_hot(v::AbstractVector{<:CategoricalValue}, ref) = map(v) do c
    MMI.int(c) == ref
end

function MMI.transform(transformer::OneHotEncoder, fitresult, X)
    features = Tables.schema(X).names     # tuple not vector

    d = fitresult.ref_name_pairs_given_feature

    # check the features match the fit result
    all(e -> e in fitresult.all_features, features) ||
        error("Attempting to transform table with feature "*
              "names not seen in fit. ")
    new_features = Symbol[]
    new_cols = [] # not Vector[] !!
    features_to_be_transformed = keys(d)
    for ftr in features
        col = MMI.selectcols(X, ftr)
        if ftr in features_to_be_transformed
            Set(fitresult.fitted_levels_given_feature[ftr]) == Set(MMI.classes(col)) ||
            error("Found category level mismatch in feature `$(ftr)`. "*
            "Consider using `levels!` to ensure fitted and transforming "*
            "features have the same category levels.")
            append!(new_features, last.(d[ftr]))
            pairs = d[ftr]
            refs = first.(pairs)
            names = last.(pairs)
            cols_to_add = map(refs) do ref
                float.(_hot(col, ref))
            end
            append!(new_cols, cols_to_add)
        else
            push!(new_features, ftr)
            push!(new_cols, col)
        end
    end
    named_cols = NamedTuple{tuple(new_features...)}(tuple(new_cols)...)
    return MMI.table(named_cols, prototype=X)
end


## CONTINUOUS_ENCODING

"""
    ContinuousEncoder(one_hot_ordered_factors=false, drop_last=false)

Unsupervised model for arranging all features (columns) of a table to
have `Continuous` element scitype, by applying the following protocol
to each feature `ftr`:

- If `ftr` is already `Continuous` retain it.

- If `ftr` is `Multiclass`, one-hot encode it.

- If `ftr` is `OrderedFactor`, replace it with `coerce(ftr,
  Continuous)` (vector of floating point integers), unless
  `ordered_factors=false` is specified, in which case one-hot encode
  it.

- If `ftr` is `Count`, replace it with `coerce(ftr, Continuous)`.

- If `ftr` is of some other element scitype, or was not observed in
  fitting the encoder, drop it from the table.

If `drop_last=true` is specified, then one-hot encoding always drops
the last class indicator column.

*Warning:* This transformer assumes that `levels(col)` for any
`Multiclass` or `OrderedFactor` column is the same in new data being
transformed as it is in the data used to fit the transformer.

### Example

```julia
X = (name=categorical(["Danesh", "Lee", "Mary", "John"]),
     grade=categorical([:A, :B, :A, :C], ordered=true),
     height=[1.85, 1.67, 1.5, 1.67],
     n_devices=[3, 2, 4, 3],
     comments=["the force", "be", "with you", "too"])
schema(X)

??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
??? _.names   ??? _.types                         ??? _.scitypes       ???
??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
??? name      ??? CategoricalValue{String,UInt32} ??? Multiclass{4}    ???
??? grade     ??? CategoricalValue{Symbol,UInt32} ??? OrderedFactor{3} ???
??? height    ??? Float64                         ??? Continuous       ???
??? n_devices ??? Int64                           ??? Count            ???
??? comments  ??? String                          ??? Textual          ???
??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
_.nrows = 4

cont = ContinuousEncoder(drop_last=true);
mach = fit!(machine(cont, X))
transform(mach, X) |> schema

?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
??? _.names      ??? _.types ??? _.scitypes ???
?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
??? name__Danesh ??? Float64 ??? Continuous ???
??? name__John   ??? Float64 ??? Continuous ???
??? name__Lee    ??? Float64 ??? Continuous ???
??? grade        ??? Float64 ??? Continuous ???
??? height       ??? Float64 ??? Continuous ???
??? n_devices    ??? Float64 ??? Continuous ???
?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
_.nrows = 4
```

"""
@with_kw_noshow mutable struct ContinuousEncoder <: Unsupervised
    drop_last::Bool                = false
    one_hot_ordered_factors::Bool  = false
end

function MMI.fit(transformer::ContinuousEncoder, verbosity::Int, X)

    # what features can be converted and therefore kept?
    s = schema(X)
    features = s.names
    scitypes = s.scitypes
    Convertible = Union{Continuous, Finite, Count}
    feature_scitype_tuples = zip(features, scitypes) |> collect
    features_to_keep  =
        first.(filter(t -> last(t) <: Convertible, feature_scitype_tuples))
    features_to_be_dropped = setdiff(collect(features), features_to_keep)

    if verbosity > 0
        if !isempty(features_to_be_dropped)
            @info "Some features cannot be replaced with "*
            "`Continuous` features and will be dropped: "*
            "$features_to_be_dropped. "
        end
    end

    # fit the one-hot encoder:
    hot_encoder =
        OneHotEncoder(ordered_factor=transformer.one_hot_ordered_factors,
                      drop_last=transformer.drop_last)
    hot_fitresult, _, hot_report = MMI.fit(hot_encoder, verbosity - 1, X)

    new_features = setdiff(hot_report.new_features, features_to_be_dropped)

    fitresult = (features_to_keep=features_to_keep,
                 one_hot_encoder=hot_encoder,
                 one_hot_encoder_fitresult=hot_fitresult)

    # generate the report:
    report = (features_to_keep=features_to_keep,
              new_features=new_features)

    cache = nothing

    return fitresult, cache, report

end

MMI.fitted_params(::ContinuousEncoder, fitresult) = fitresult

function MMI.transform(transformer::ContinuousEncoder, fitresult, X)

    features_to_keep, hot_encoder, hot_fitresult = values(fitresult)

    # dump unseen or untransformable features:
    selector = FeatureSelector(features=features_to_keep)
    selector_fitresult, _, _ = MMI.fit(selector, 0, X)
    X0 = transform(selector, selector_fitresult, X)

    # one-hot encode:
    X1 = transform(hot_encoder, hot_fitresult, X0)

    # convert remaining to continuous:
    return coerce(X1, Count=>Continuous, OrderedFactor=>Continuous)

end

##
## Metadata for all built-in transformers
##

metadata_pkg.(
    (FeatureSelector, UnivariateStandardizer,
     UnivariateDiscretizer, Standardizer,
     UnivariateBoxCoxTransformer, UnivariateFillImputer,
     OneHotEncoder, FillImputer, ContinuousEncoder,
     UnivariateTimeTypeToContinuous),
    name       = "MLJModels",
    uuid       = "d491faf4-2d78-11e9-2867-c94bc002c0b7",
    url        = "https://github.com/alan-turing-institute/MLJModels.jl",
    julia      = true,
    license    = "MIT",
    is_wrapper = false)

metadata_model(UnivariateFillImputer,
    input = Union{AbstractVector{<:Union{Continuous,Missing}},
                  AbstractVector{<:Union{Count,Missing}},
                  AbstractVector{<:Union{Finite,Missing}}},
    output = Union{AbstractVector{<:Continuous},
                  AbstractVector{<:Count},
                  AbstractVector{<:Finite}},
    descr = UNIVARIATE_FILL_IMPUTER_DESCR,
    path  = "MLJModels.UnivariateFillImputer")

metadata_model(FillImputer,
    input   = Table,
    output  = Table,
    weights = false,
    descr   = FILL_IMPUTER_DESCR,
    path    = "MLJModels.FillImputer")

metadata_model(FeatureSelector,
    input   = Table,
    output  = Table,
    weights = false,
    descr   = FEATURE_SELECTOR_DESCR,
    path    = "MLJModels.FeatureSelector")

metadata_model(UnivariateDiscretizer,
    input   = AbstractVector{<:Continuous},
    output  = AbstractVector{<:OrderedFactor},
    weights = false,
    descr   = UNIVARIATE_DISCR_DESCR,
    path    = "MLJModels.UnivariateDiscretizer")

metadata_model(UnivariateStandardizer,
    input   = AbstractVector{<:Infinite},
    output  = AbstractVector{Continuous},
    weights = false,
    descr   = UNIVARIATE_STD_DESCR,
    path    = "MLJModels.UnivariateStandardizer")

metadata_model(Standardizer,
    input   = Union{Table, AbstractVector{<:Continuous}},
    output  = Union{Table, AbstractVector{<:Continuous}},
    weights = false,
    descr   = STANDARDIZER_DESCR,
    path    = "MLJModels.Standardizer")

metadata_model(UnivariateBoxCoxTransformer,
    input   = AbstractVector{Continuous},
    output  = AbstractVector{Continuous},
    weights = false,
    descr   = UNIVARIATE_BOX_COX_DESCR,
    path    = "MLJModels.UnivariateBoxCoxTransformer")

metadata_model(OneHotEncoder,
    input   = Table,
    output  = Table,
    weights = false,
    descr   = ONE_HOT_DESCR,
    path    = "MLJModels.OneHotEncoder")

metadata_model(ContinuousEncoder,
    input   = Table,
    output  = Table(Continuous),
    weights = false,
    descr   = CONTINUOUS_ENCODER_DESCR,
    path    = "MLJModels.ContinuousEncoder")

metadata_model(UnivariateTimeTypeToContinuous,
    input   = AbstractVector{<:ScientificTimeType},
    output  = AbstractVector{Continuous},
    weights = false,
    descr   = UNIVARIATE_TIME_TYPE_TO_CONTINUOUS,
    path    = "MLJModels.UnivariateTimeTypeToContinuous")
