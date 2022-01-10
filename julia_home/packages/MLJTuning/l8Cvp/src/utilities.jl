# extract items useful for plotting from the history generated by the
# fallback `result` method:
function plotting_report(fields, scales, history)

    n_models = length(history)
    n_parameters = length(fields)

    A = Array{Any}(undef, (n_models, n_parameters))
    measurements = Vector{Float64}(undef, n_models)

    for j in eachindex(history)
        entry = history[j]
        A[j,:] = [recursive_getproperty(entry.model, fld) for fld in fields]
        measurements[j] = entry.measurement[1]
    end

    return plotting=(parameter_names=string.(fields) |> collect,
                     parameter_scales=scales |> collect,
                     parameter_values = A,
                     measurements = measurements)

end

# return a named tuple with some keys removed:
function delete(nt::NamedTuple, target_keys...)
    zipped = tuple(zip(keys(nt), values(nt))...)
    filtered = filter(zipped) do tup
        !(tup[1] in target_keys)
    end
    return (; filtered...)
end

signature(measure) =
    if orientation(measure) == :loss
        1
    elseif orientation(measure) == :score
        -1
    else
        0
    end