cd(@__DIR__)

# Setup
using Pkg
!isdir("ImageMagick-Test") && mkdir("ImageMagick-Test")
Pkg.activate(joinpath(@__DIR__,"ImageMagick-Test"))
Pkg.develop(PackageSpec(path="/Users/ian/Documents/GitHub/FileIO.jl"))
Pkg.add("ImageMagick")
Pkg.instantiate()

!isdir("QuartzImageIO-Test") && mkdir("QuartzImageIO-Test")
Pkg.activate(joinpath(@__DIR__,"QuartzImageIO-Test"))
Pkg.develop(PackageSpec(path="/Users/ian/Documents/GitHub/FileIO.jl"))
Pkg.add("QuartzImageIO")
Pkg.instantiate()

!isdir("PNGFiles-Test") && mkdir("PNGFiles-Test")
Pkg.activate(joinpath(@__DIR__,"PNGFiles-Test"))
Pkg.develop(PackageSpec(path="/Users/ian/Documents/GitHub/FileIO.jl"))
Pkg.develop(PackageSpec(path="/Users/ian/Documents/GitHub/PNGFiles.jl"))
Pkg.instantiate()

Pkg.activate()

corebench = raw"""
    using BenchmarkTools
    tfirst = @elapsed begin
        using FileIO
        FileIO.save("tst.png", x)
    end
    rm("tst.png")
    println("Time to first save (including save): $(round(tfirst, digits=3)) seconds")

    print("Save @btime:")
    @btime FileIO.save("tst.png", x) teardown=(rm("tst.png"))
    FileIO.save("tst.png", x)

    print("Load @btime:")
    @btime FileIO.load("tst.png")
    rm("tst.png")
    println("")
    """

for imgstr in ["rand(RGB{N0f8}, 244, 244)", "rand(RGB{N0f8}, 20000, 4000)"]
    setup = """
    using ImageCore
    x = $imgstr
    """

    @info "Testing with $imgstr image -----------------------------"

    for pkgname  in ["ImageMagick", "QuartzImageIO", "PNGFiles"]
        @info "Testing FileIO with $pkgname (dedicated environment, new instance)"
        cd(joinpath(@__DIR__,"$pkgname-Test"))
        run(`julia --project=Project.toml --color=yes -e $(string(setup, corebench))`)
    end
end
