@testset "JLSOFile" begin

    withenv("JLSO_IMAGE" => "busybox") do
        jlso = JLSOFile(:data => "the image env variable is set")
        @test jlso.image == "busybox"
    end

    # Reset the cached image for future tests
    JLSO._CACHE[:IMAGE] = ""

    @testset "$(fmt): $k" for fmt in (:bson, :julia_serialize), (k, v) in datas
        jlso = JLSOFile(k => v; format=fmt, compression=:none)
        io = IOBuffer()
        bytes = fmt === :bson ? bson(io, Dict("object" => v)) : serialize(io, v)
        expected = take!(io)

        @test jlso.objects[k] == expected
    end

    @testset "kwarg constructor" begin
        jlso = JLSOFile(; a=collect(1:10), b="hello")
        @test jlso[:b] == "hello"
        @test haskey(jlso.manifest, "BSON")
    end

    @testset "no-arg constructor" begin
        jlso = JLSOFile()
        @test jlso isa JLSOFile
        @test isempty(jlso.objects)
        @test haskey(jlso.manifest, "BSON")
    end
end

@testset "unknown format" begin
    @test_throws(
        LOGGER,
        MethodError,
        JLSOFile("String" => "Hello World!", format=:unknown)
    )
end

@testset "show" begin
    jlso = JLSOFile(:string => datas[:String])
    expected = string(
        "JLSOFile([data]; version=v\"2.0.0\", julia=v\"$VERSION\", ",
        "format=:julia_serialize, image=\"\")"
    )
    @test sprint(show, jlso) == sprint(print, jlso)
end

@testset "activate" begin
    jlso = JLSOFile(:string => datas[:String])
    mktempdir() do d
        Pkg.activate(jlso, d) do
            @show Base.active_project()
        end

        @test ispath(joinpath(d, "Project.toml"))
        @test ispath(joinpath(d, "Manifest.toml"))

        @test Pkg.TOML.parsefile(joinpath(d, "Project.toml")) == jlso.project
        @test Pkg.TOML.parsefile(joinpath(d, "Manifest.toml")) == jlso.manifest
    end
end

@testset "keys/haskey" begin
    jlso = JLSOFile(:string => datas[:String])
    @test collect(keys(jlso)) == [:string]
    @test haskey(jlso, :string)
    @test !haskey(jlso, :other)
end

@testset "get/get!" begin
    v = datas[:String]
    jlso = JLSOFile(:str => v)
    @test get(jlso, :str, "fail") == v
    @test get!(jlso, :str, "fail") == v

    @test get(jlso, :other, v) == v
    @test !haskey(jlso, :other)

    @test get!(jlso, :other, v) == v
    @test jlso[:other] == v

    # key must be a Symbol
    @test get(jlso, "str", 999) == 999
    @test_throws MethodError get!(jlso, "str", 999)
end
