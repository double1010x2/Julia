module Registry

# for this module
import Pkg
import Pkg.TOML
using InteractiveUtils

# for testings decoding of metadata:
import MLJBase: Found, Continuous, Finite, Infinite
import MLJBase: OrderedFactor, Count, Multiclass, Binary

const srcdir = dirname(@__FILE__) # the directory containing this file
const environment_path = joinpath(srcdir, "..")

## METHODS TO GENERATE METADATA AND WRITE TO ARCHIVE

function finaltypes(T::Type)
    s = InteractiveUtils.subtypes(T)
    if isempty(s)
        return [T, ]
    else
        return reduce(vcat, [finaltypes(S) for S in s])
    end
end

const project_toml = joinpath(srcdir, "../Project.toml")
const packages = map(Symbol,
                     keys(TOML.parsefile(project_toml)["deps"])|>collect)
push!(packages, :MLJModels)
filter!(packages) do pkg
    !(pkg in (:InteractiveUtils, :Pkg, :MLJModelInterface))
end

const package_import_commands =  [:(import $pkg) for pkg in packages]

macro update()
    mod = __module__
    _update(mod, false)
end

macro update(ex)
    mod = __module__
    test_env_only = eval(ex)
    test_env_only isa Bool || "b in @update(b) must be Bool. "
    _update(mod, test_env_only)
end

function _update(mod, test_env_only)

    test_env_only && @info "Testing registry environment only. "

    program1 = quote
        @info "Packages to be searched for model implementations:"
        for pkg in $packages
            println(pkg)
        end
        using Pkg
        Pkg.activate($environment_path)
        @info "resolving registry environment..."
        Pkg.resolve()
    end

    program2 = quote

        @info "Instantiating registry environment..."
        Pkg.instantiate()

        @info "Loading registered packages..."
        import MLJBase
        import MLJModels
        using Pkg.TOML

        # import the packages
        $(Registry.package_import_commands...)

        @info "Generating model metadata..."

        modeltypes = MLJModels.Registry.finaltypes(MLJBase.Model)
        filter!(modeltypes) do T
            !isabstracttype(T) && !MLJBase.is_wrapper(T)
        end

        # generate and write to file the model metadata:
        api_packages = string.(MLJModels.Registry.packages)
        meta_given_package = Dict()

        for M in modeltypes
            _info = MLJModels.info_dict(M)
            pkg = _info[:package_name]
            path = _info[:load_path]
            api_pkg = split(path, '.') |> first
            pkg in ["unknown",] &&
                @warn "$M `package_name` or `load_path` is \"unknown\")"
            modelname = _info[:name]
            api_pkg in api_packages ||
                error("Bad `load_path` trait for $M: "*
                      "$api_pkg not a registered package. ")
            haskey(meta_given_package, pkg) ||
                (meta_given_package[pkg] = Dict())
            haskey(meta_given_package, modelname) &&
                error("Encountered multiple model names for "*
                      "`package_name=$pkg`")
            meta_given_package[pkg][modelname] = _info
                println(M, "\u2714 ")
        end
        print("\r")

        open(joinpath(MLJModels.Registry.srcdir, "../Metadata.toml"), "w") do file
            TOML.print(file, MLJModels.encode_dic(meta_given_package))
        end

        # generate and write to file list of models for each package:
        models_given_pkg = Dict()
        for pkg in keys(meta_given_package)
            models_given_pkg[pkg] = collect(keys(meta_given_package[pkg]))
        end
        open(joinpath(MLJModels.Registry.srcdir, "../Models.toml"), "w") do file
            TOML.print(file, models_given_pkg)
        end

        :(println("Local Metadata.toml updated."))

    end

    mod.eval(program1)
    test_env_only || mod.eval(program2)

    println("\n You can check the registry by running "*
            "`MLJModels.check_registry() but may need to force "*
            "recompilation of MLJModels.\n\n"*
            "You can safely ignore \"conflicting import\" warnings. ")

    true
end

end # module
