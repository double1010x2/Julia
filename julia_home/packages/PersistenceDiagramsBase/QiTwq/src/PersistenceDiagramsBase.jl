module PersistenceDiagramsBase

using Compat
using Tables

export PersistenceDiagram,
    PersistenceInterval,
    birth,
    death,
    persistence,
    midlife,
    dim,
    threshold,
    birth_simplex,
    death_simplex,
    representative

include("intervals.jl")
include("diagrams.jl")
include("tables.jl")

end
