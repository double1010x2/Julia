@inline function _AudzeEglaisDist(LHC,periodic_ae,ae_power,periodic_n)
    n,d = size(LHC)
    dist = 0.0

    #l-2 norm of distances between all (unique) points to the power of ae_power
    for i = 2:n
        for j = 1:i-1
            dist_tmp = 0.0
            for k = 1:d
                if periodic_ae
                    @inbounds dist_comp = abs(LHC[i,k]-LHC[j,k])
                    dist_comp = min(dist_comp,periodic_n-dist_comp)
                else
                    @inbounds dist_comp = LHC[i,k]-LHC[j,k]
                end
                dist_tmp += dist_comp^2
            end
            #This if statement improves a performance regression when running the entire LHCoptim
            #with the standard power of 2
            if ae_power != 2                    
                dist += 1/dist_tmp^(ae_power/2)
            else
                dist += 1/dist_tmp
            end
        end
    end
    output = 1/dist
    return output
end

function _AudzeEglaisObjective(dim::Continuous,LHC,periodic_ae,ae_power,periodic_n)
    output = _AudzeEglaisDist(LHC,periodic_ae,ae_power,periodic_n)    
    return output
end

function _AudzeEglaisObjective(dim::Categorical,LHC,periodic_ae,ae_power,periodic_n)
    output = _AudzeEglaisDist(LHC,periodic_ae,ae_power,periodic_n)     
    output == Inf ? 0 : output    
end

"""
    function AudzeEglaisObjective!(LHC::T) where T <: AbstractArray
Return the scalar which should be maximized when using the Audze-Eglais
distance as the objective function. Note this is the inverse of the typical
Audze-Eglais distance which normally is minimized.
"""
function AudzeEglaisObjective(LHC::T; dims::Array{V,1} =[Continuous() for i in 1:size(LHC,2)],
                                            interSampleWeight::Float64=1.0,periodic_ae::Bool=false,ae_power::Union{Int,Float64}=2,
                                            periodic_n::Int = size(LHC,1)
                                            ) where T <: AbstractArray where V <: LHCDimension
    
    out = 0.0

    #Compute the objective function among all points
    out += _AudzeEglaisObjective(Continuous(),LHC,periodic_ae,ae_power,periodic_n)*interSampleWeight

    #Compute the objective function within each categorical dimension
    categoricalDimInds = findall(x->typeof(x)==Categorical,dims)
    for i in categoricalDimInds
        for j = 1:dims[i].levels
            subLHC = @view LHC[LHC[:,i] .== j,:] 
            out += _AudzeEglaisObjective(dims[i],subLHC,periodic_ae,ae_power,periodic_n)*dims[i].weight
        end
    end

    return out
end

# Remove depwarning in release 2.x.x
function AudzeEglaisObjective!(dist,LHC::T; dims::Array{V,1} =[Continuous() for i in 1:size(LHC,2)],
                                            interSampleWeight::Float64=1.0,periodic_ae::Bool=false,ae_power::Union{Int,Float64}=2,
                                            periodic_n::Int = size(LHC,1)
                                            ) where T <: AbstractArray where V <: LHCDimension
    @warn "AudzeEglaisObjective!(dist,LHC) is deprecated and does not differ from AudzeEglaisObjective(LHC)"
    AudzeEglaisObjective(LHC; dims = dims, interSampleWeight = interSampleWeight, periodic_ae=periodic_ae, ae_power=ae_power, periodic_n=periodic_n)
end