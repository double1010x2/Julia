# Overview of StructArrays.jl

This package introduces the type `StructArray` which is an `AbstractArray` whose elements are `struct` (for example `NamedTuples`,  or `ComplexF64`, or a custom user defined `struct`). While a `StructArray` iterates `structs`, the layout is column based (meaning each field of the `struct` is stored in a separate `Array`, and `struct` entries of a StructArray are constructed on-the-fly). 

`Base.getproperty` or the dot syntax can be used to access columns, whereas rows can be accessed with `getindex`. 

The package was largely inspired by the `Columns` type in [IndexedTables](https://github.com/JuliaComputing/IndexedTables.jl) which it now replaces.

## Collection and initialization

One can create a `StructArray` by providing the struct type and a tuple or NamedTuple of field arrays:
```julia
julia> struct Foo{T}
       a::T
       b::T
       end

julia> x = StructArray{Foo}((rand(2,2), rand(2,2)))
2×2 StructArray(::Matrix{Float64}, ::Matrix{Float64}) with eltype Foo:
 Foo{Float64}(0.702413, 0.416194)  Foo{Float64}(0.520032, 0.0856553)
 Foo{Float64}(0.701297, 0.977394)  Foo{Float64}(0.451654, 0.258264)

julia> x = StructArray{Foo}((a=rand(2,2), b=rand(2,2)))
2×2 StructArray(::Matrix{Float64}, ::Matrix{Float64}) with eltype Foo:
 Foo{Float64}(0.702413, 0.416194)  Foo{Float64}(0.520032, 0.0856553)
 Foo{Float64}(0.701297, 0.977394)  Foo{Float64}(0.451654, 0.258264) 
```
If a struct is not specified, a StructArray of Tuple or NamedTuple type will be created
```julia 
julia> x = StructArray((rand(2,2), rand(2,2)))
2×2 StructArray(::Matrix{Float64}, ::Matrix{Float64}) with eltype Tuple{Float64, Float64}:
 (0.754912, 0.803434)  (0.341105, 0.904933)
 (0.348966, 0.550496)  (0.199761, 0.511388)

julia> x = StructArray((x = rand(2,2), y = rand(2,2)))
2×2 StructArray(::Matrix{Float64}, ::Matrix{Float64}) with eltype NamedTuple{(:x, :y), Tuple{Float64, Float64}}:
 (x = 0.486, y = 0.00588886)   (x = 0.97401, y = 0.834173)
 (x = 0.394493, y = 0.192362)  (x = 0.107698, y = 0.694548)
```


One can also create a `StructArray` from an iterable of structs without creating an intermediate `Array`:

```julia
julia> StructArray(log(j+2.0*im) for j in 1:10)
10-element StructArray(::Array{Float64,1}, ::Array{Float64,1}) with eltype Complex{Float64}:
 0.8047189562170501 + 1.1071487177940904im
 1.0397207708399179 + 0.7853981633974483im
 1.2824746787307684 + 0.5880026035475675im
 1.4978661367769954 + 0.4636476090008061im
  1.683647914993237 + 0.3805063771123649im
 1.8444397270569681 + 0.3217505543966422im
  1.985145956776061 + 0.27829965900511133im
 2.1097538525880535 + 0.24497866312686414im
 2.2213256282451583 + 0.21866894587394195im
 2.3221954495706862 + 0.19739555984988078im
```

Another option is to create an uninitialized `StructArray` and then fill it with data. Just like in normal arrays, this is done with the `undef` syntax:

```julia
julia> s = StructArray{ComplexF64}(undef, 2, 2)
2×2 StructArray(::Array{Float64,2}, ::Array{Float64,2}) with eltype Complex{Float64}:
 6.91646e-310+6.91646e-310im  6.91646e-310+6.91646e-310im
 6.91646e-310+6.91646e-310im  6.91646e-310+6.91646e-310im

julia> rand!(s)
2×2 StructArray(::Array{Float64,2}, ::Array{Float64,2}) with eltype Complex{Float64}:
 0.680079+0.874437im  0.625239+0.737254im
  0.92407+0.929336im  0.267358+0.804478im
```
## Using custom array types

StructArrays supports using custom array types. It is always possible to pass field arrays of a custom type. The "custom array of structs to struct of custom arrays" transformation will use the `similar` method of the custom array type. This can be useful when working on the GPU for example:

```julia
julia> using StructArrays, CuArrays

julia> a = CuArray(rand(Float32, 10));

julia> b = CuArray(rand(Float32, 10));

julia> StructArray{ComplexF32}((a, b))
10-element StructArray(::CuArray{Float32,1}, ::CuArray{Float32,1}) with eltype Complex{Float32}:
  0.19555175f0 + 0.9604322f0im
  0.68348145f0 + 0.5778245f0im
  0.69664395f0 + 0.79825306f0im
 0.118531585f0 + 0.3031248f0im
  0.80057466f0 + 0.8964418f0im
  0.63772964f0 + 0.2923274f0im
  0.65374136f0 + 0.7932533f0im
   0.6043732f0 + 0.65964353f0im
   0.1106627f0 + 0.090207934f0im
    0.707458f0 + 0.1700114f0im

julia> c = CuArray(rand(ComplexF32, 10));

julia> StructArray(c)
10-element StructArray(::Array{Float32,1}, ::Array{Float32,1}) with eltype Complex{Float32}:
  0.7176411f0 + 0.864058f0im
   0.252609f0 + 0.14824867f0im
 0.26842773f0 + 0.9084332f0im
 0.33128333f0 + 0.5106474f0im
  0.6509278f0 + 0.87059164f0im
  0.9522146f0 + 0.053706646f0im
   0.899577f0 + 0.63242567f0im
   0.325814f0 + 0.59225655f0im
 0.56267905f0 + 0.21927536f0im
 0.49719965f0 + 0.754143f0im
```

If you already have your data in a `StructArray` with field arrays of a given format (say plain `Array`) you can change them with `replace_storage`:

```julia
julia> s = StructArray([1.0+im, 2.0-im])
2-element StructArray(::Array{Float64,1}, ::Array{Float64,1}) with eltype Complex{Float64}:
 1.0 + 1.0im
 2.0 - 1.0im

julia> replace_storage(CuArray, s)
2-element StructArray(::CuArray{Float64,1}, ::CuArray{Float64,1}) with eltype Complex{Float64}:
 1.0 + 1.0im
 2.0 - 1.0im
```

## Lazy row iteration

StructArrays also provides a `LazyRow` wrapper for lazy row iteration. `LazyRow(t, i)` does not materialize the i-th row but returns a lazy wrapper around it on which `getproperty` does the correct thing. This is useful when the row has many fields only some of which are necessary. It also allows changing columns in place.

```julia
julia> t = StructArray((a = [1, 2], b = ["x", "y"]));

julia> LazyRow(t, 2).a
2

julia> LazyRow(t, 2).a = 123
123

julia> t
2-element StructArray(::Array{Int64,1}, ::Array{String,1}) with eltype NamedTuple{(:a, :b),Tuple{Int64,String}}:
 (a = 1, b = "x")
 (a = 123, b = "y")
```

To iterate in a lazy way one can simply iterate `LazyRows`:

```julia
julia> map(t -> t.b ^ t.a, LazyRows(t))
2-element Array{String,1}:
 "x"
 "yy"
```

## Applying a function on each field array

```julia
julia> struct Foo
       a::Int
       b::String
       end

julia> s = StructArray([Foo(11, "a"), Foo(22, "b"), Foo(33, "c"), Foo(44, "d"), Foo(55, "e")]);

julia> s
5-element StructArray(::Vector{Int64}, ::Vector{String}) with eltype Foo:
 Foo(11, "a")
 Foo(22, "b")
 Foo(33, "c")
 Foo(44, "d")
 Foo(55, "e")

julia> StructArrays.foreachfield(v -> deleteat!(v, 3), s)

julia> s
4-element StructArray(::Vector{Int64}, ::Vector{String}) with eltype Foo:
 Foo(11, "a")
 Foo(22, "b")
 Foo(44, "d")
 Foo(55, "e")
```

