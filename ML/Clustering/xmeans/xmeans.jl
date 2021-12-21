#========================================================================
=   X-means algorithm                                                   =
=   Ref:                                                                =
=       (1) https://github.com/KazuhisaFujita/X-means                   =
=       (2) Extending K-means with Efficient Estimation                 = 
=            of the Number of Clusters, D. Pelleg and A. Moore (2000)   =
========================================================================#

# Macro: for print loading package
macro used(pka::String)
    eval(Meta.parse("using $(pka)"))
    println("Loading package $(pka)")
end

@used "Clustering"
@used "ArgParse"
@used "Debugger"
@used "Dates"
@used "RDatasets"
@used "MLJ";        const mlj   = MLJ
@used "DataFrames"; const df    = DataFrames
@used "Plots";      const pl    = Plots
default(show=true)

# Macro: Print time information 
macro timePrint(flag::String)
    t = Dates.now()
    Y = Dates.year(t)
    M = Dates.month(t)
    D = Dates.day(t)
    h = Dates.hour(t)
    m = Dates.minute(t)
    s = Dates.second(t)
    println("[$Y-$M-$D $h:$m:$s] - $flag")
end

global eps_g = nextfloat(0.0)
function loglikelihood(r::Int64, rn::Int64, var::Float64, m::Int64, k::Int64)
    l1 = - rn / 2.0 * log(2 * pi+eps_g)
    l2 = - rn * m / 2.0 * log(var+eps_g)
    l3 = - (rn - k) / 2.0
    l4 = rn * log(rn+eps_g)
    l5 = - rn * log(r+eps_g)
    return l1 + l2 + l3 + l4 + l5
end

#struct XMeansResult{C<:AbstractMatrix{<:AbstractFloat},D<:Real,WC<:Real}
struct XMeansResult{K<:Int64, L<:Vector{Int64}, C<:Matrix{Float64}}
    k::K
    labels::L
    centers::C
end

function XMeans(X::Matrix{<:Real},
                kmin::Int64=1,
                kmax::Int64=50,
                maxiter::Int64=100;
                kwargs...
                )

    k     = kmin
    M     = size(X)[end]
    num   = size(X)[begin]
    while true 
        ok = k
        km = kmeans(X', k; maxiter=maxiter)
        labels = km.assignments
        m  = km.centers'

        # calc BIC
        p  = M + 1
        obic = zeros(Float64, k)

        for i in range(1, k)
            rn = length(filter(x -> x==i, labels))
            var = (sum((X[labels .== i,:] .- m[[i],:]).^2)+eps_g) / (rn - 1 + eps_g)
            obic[i] = loglikelihood(rn, rn, var, M, 1) - p/2.0*log(rn + eps_g)
        end

        sk = 2
        nbic = zeros(Float64, k)
        addk = 0::Int64

        for i in collect(1:k)
            ci = X[labels .== i,:]
            r  = length(filter( x -> x==i, labels))
            km = kmeans(ci', sk; maxiter=maxiter)
            ci_labels = km.assignments
            sm = km.centers'

            for l in collect(1:sk)
                rn = length(filter(x -> x==l, ci_labels))
                var = (sum((ci[ci_labels .== l,:] .- sm[[l],:]).^2) + eps_g) / (rn-sk + eps_g)
                nbic[i] += loglikelihood(r, rn, var, M, sk)
            end

            p = sk * (M+1)
            nbic[i] -= p/2.0*log(r+eps_g)

            if obic[i] < nbic[i]
                addk += 1::Integer
            end
        end

        k += addk

        if ok == k || k > kmax
            break;
        end
    end

    #Calculate labels and centroids
    km = kmeans(X', k; maxiter=maxiter)
#    results = XMeansResult(k, km.assignments, km.centers') 
    labels = km.assignments
    k = k
    m = km.centers
    return XMeansResult(k, labels, m)

end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--kmin", "-k"
            help = "k min value for xmenas"
            arg_type = Int
            default = 1
        "--kmax", "-K"
            help = "k max value for xmeans"
            arg_type = Int
            default = 50
        "--maxiter", "-M"
            help = "maxiter for kmeans in xmeans"
            arg_type = Int
            default = 100
        "--data_type", "-D"
            help = "Data type for xmeans, iris/make_blobs"
            arg_type = String
            default  = "make_blobs" 
#        "--flag1"
#            help = "an option without argument, i.e. a flag"
#            action = :store_true
#        "arg1"
#            help = "a positional argument"
#            required = true
    end

    return parse_args(s)
end

function main()

    _args = parse_commandline()

    @timePrint "@XMeans : Loading Data by make_blobs!!!"
    #====================  
    =  MLJ make_blobs   =    
    ====================#
    if _args["data_type"] == "make_blobs"
        X, y = mlj.make_blobs(1000, 2; centers=5, cluster_std=[1.0, 1.0, 1.0, 1.0, 1.0])
        features = Matrix(df.DataFrame(X))
    end

    #================  
    =  iris data    =      
    ================# 
    if _args["data_type"] == "iris"
        iris = dataset("datasets", "iris");
        features = collect(Matrix(iris[:, 1:2]));
    end

    @timePrint "@XMeans : Clustering start !!!"
    results = XMeans(features; kmin=_args["kmin"], kmax=_args["kmax"], maxiter=_args["maxiter"])
    println("@XMeans - k=$(results.k)")

    @timePrint "@XMeans : Display plot !!!"
    p = pl.scatter(features[:,1], features[:,2], marker_z=results.labels,
        color=:lightrainbow, legend=false, title="xmeans results")
    pl.scatter!(results.centers'[:,1], results.centers'[:,2], color="black", markersize=6, marker=:hex)
    pl.plot!(xlabel="Dim1", ylabel="Dim2")
    display(p)
    sleep(30)

    #################################
    #                               #
    #   debug argparse output       #
    #                               #
    #################################
    #parsed_args = parse_commandline()
    #println("Parsed args:")
    #for (arg,val) in parsed_args
    #    println("  $arg  =>  $val")
    #end
end



# same as python __name__ == __main__
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

"""
[kmeans results from source file]
struct KmeansResult{C<:AbstractMatrix{<:AbstractFloat},D<:Real,WC<:Real} <: ClusteringResult
    centers::C                 # cluster centers (d x k)
    assignments::Vector{Int}   # assignments (n)
    costs::Vector{D}           # cost of the assignments (n)
    counts::Vector{Int}        # number of points assigned to each cluster (k)
    wcounts::Vector{WC}        # cluster weights (k)
    totalcost::D               # total cost (i.e. objective)
    iterations::Int            # number of elapsed iterations
    converged::Bool            # whether the procedure converged
end
"""