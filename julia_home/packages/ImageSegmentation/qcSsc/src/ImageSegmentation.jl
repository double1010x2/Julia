module ImageSegmentation

import Base: show

# To keep the symbols in the namespace maintainble, we try to be explicit on symbols
# importing here, especially for external symbols developed outside of JuliaImages
using LinearAlgebra, Statistics
using DataStructures, StaticArrays, ImageCore, ImageFiltering, RegionTrees, Distances, StaticArrays, Clustering
using Graphs: Edge, AbstractEdge, nv, neighbors, vertices
using Graphs: add_vertices!, add_edge!
using MetaGraphs: MetaGraph
using ImageMorphology: erode
using SimpleWeightedGraphs: SimpleWeightedGraph
using ImageCore.ColorVectorSpace: MathTypes
import Clustering: kmeans, fuzzy_cmeans

const PairOrTuple{K,V} = Union{Pair{K,V},Tuple{K,V}}

include("compat.jl")
include("core.jl")
include("region_growing.jl")
include("felzenszwalb.jl")
include("fast_scanning.jl")
include("flood_fill.jl")
include("watershed.jl")
include("region_merging.jl")
include("meanshift.jl")
include("clustering.jl")
include("merge_segments.jl")
include("deprecations.jl")

export
    #accessor methods
    labels_map,
    segment_labels,
    segment_pixel_count,
    segment_mean,

    # methods
    seeded_region_growing,
    unseeded_region_growing,
    felzenszwalb,
    fast_scanning,
    fast_scanning!,
    flood,
    flood_fill!,
    watershed,
    hmin_transform,
    region_adjacency_graph,
    remove_segment,
    remove_segment!,
    prune_segments,
    region_tree,
    region_splitting,
    meanshift,
    kmeans,
    fuzzy_cmeans,
    merge_segments,

    # types
    SegmentedImage,
    ImageEdge

@deprecate rem_segment  remove_segment
@deprecate rem_segment! remove_segment!

end # module
