# CoupledFields.jl

## Summary

A julia package for working with coupled fields. This is work in progress. 
The main function `gradvecfield` calculates the gradient vector or gradient matrix for each instance of the coupled fields.

For π = g(πΏ), `CoupledFields.gradvecfield([a b], X, Y, kernelpars)` returns π gradient matrices, for π random points in πΏ.
For parameters [π π]: π is a smoothness parameter, and π is a ridge parameter.

```julia
using CoupledFields
g(x,y,z) = x * exp(-x^2 - y^2 - z^2)
X = -2 .+ 4*rand(100, 3)
Y = g.(X[:,1], X[:,2], X[:,3])

 kernelpars = GaussianKP(X)
 βg = gradvecfield([0.5 -7], X, Y[:,1:1], kernelpars)
```
Also CoupledFields doesnβt require a closed-form function, it can be used if you have only the observed fields πΏ and π.


## Installation

Install using the Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:
```
pkg> add CoupledFields
```
or from the Julia prompt:
```julia
julia> using Pkg; Pkg.add("CoupledFields")
```

## Documentation

- [**LATEST**][docs-latest-url]


[docs-latest-url]: https://Mattriks.github.io/CoupledFields.jl/dev
[docs-stable-url]: https://Mattriks.github.io/CoupledFields.jl/stable

