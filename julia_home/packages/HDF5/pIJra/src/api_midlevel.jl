# This file defines midlevel api wrappers. We include name normalization for methods that are
# applicable to different hdf5 api-layers. We still try to adhere close proximity to the underlying
# method name in the hdf5-library.

"""
    HDF5.set_extent_dims(dset::HDF5.Dataset, new_dims::Dims)

Change the current dimensions of a dataset to `new_dims`, limited by
`max_dims = get_extent_dims(dset)[2]`. Reduction is possible and leads to loss of truncated data.
"""
function set_extent_dims(dset::Dataset, size::Dims)
    checkvalid(dset)
    h5d_set_extent(dset, hsize_t[reverse(size)...])
end

"""
    HDF5.set_extent_dims(dspace::HDF5.Dataspace, new_dims::Dims, max_dims::Union{Dims,Nothing} = nothing)

Change the dimensions of a dataspace `dspace` to `new_dims`, optionally with the maximum possible
dimensions `max_dims` different from the active size `new_dims`. If not given, `max_dims` is set equal
to `new_dims`.
"""
function set_extent_dims(dspace::Dataspace, size::Dims, max_dims::Union{Dims,Nothing} = nothing)
    checkvalid(dspace)
    rank = length(size)
    current_size = hsize_t[reverse(size)...]
    maximum_size = isnothing(max_dims) ? C_NULL : [reverse(max_dims .% hsize_t)...]
    h5s_set_extent_simple(dspace, rank, current_size, maximum_size)
    return nothing
end


"""
    HDF5.get_extent_dims(obj::Union{HDF5.Dataspace, HDF5.Dataset, HDF5.Attribute}) -> dims, maxdims

Get the array dimensions from a dataspace, dataset, or attribute and return a tuple of `dims` and `maxdims`.
"""
function get_extent_dims(obj::Union{Dataspace,Dataset,Attribute})
    dspace = obj isa Dataspace ? checkvalid(obj) : dataspace(obj)
    h5_dims, h5_maxdims = h5s_get_simple_extent_dims(dspace)
    # reverse dimensions since hdf5 uses C-style order
    N = length(h5_dims)
    dims = ntuple(i -> @inbounds(Int(h5_dims[N-i+1])), N)
    maxdims = ntuple(i -> @inbounds(h5_maxdims[N-i+1]) % Int, N) # allows max_dims to be specified as -1 without triggering an overflow
    obj isa Dataspace || close(dspace)
    return dims, maxdims
end


"""
    HDF5.get_chunk_offset(dataset_id, index)

Get 0-based offset of chunk from 0-based `index`. The offsets are returned in Julia's column-major order rather than hdf5 row-major order.
For a 1-based API, see `HDF5.ChunkStorage`.
"""
function get_chunk_offset(dataset_id, index)
    extent = size(dataset_id)
    chunk = get_chunk(dataset_id)
    chunk_indices = CartesianIndices(ntuple(i -> 0:extent[i]??chunk[i]-1, length(extent)))
    offset = hsize_t.(chunk_indices[index + 1].I .* chunk)
    return offset
end

"""
    HDF5.get_chunk_index(dataset_id, offset)

Get 0-based index of chunk from 0-based `offset` returned in Julia's column-major order.
For a 1-based API, see `HDF5.ChunkStorage`.
"""
function get_chunk_index(dataset_id, offset)
    extent = size(dataset_id)
    chunk = get_chunk(dataset_id)
    chunk_indices = LinearIndices(ntuple(i->0:extent[i]??chunk[i]-1, length(extent)))
    chunk_indices[(offset .?? chunk .+ 1)...] - 1
end

"""
    HDF5.get_num_chunks_per_dim(dataset_id)

Get the number of chunks in each dimension in Julia's column-major order.
"""
function get_num_chunks_per_dim(dataset_id)
    extent = size(dataset_id)
    chunk = get_chunk(dataset_id)
    return extent .?? chunk
end

"""
    HDF5.get_num_chunks(dataset_id)

Returns the number of chunks in a dataset. Equivalent to `h5d_get_num_chunks(dataset_id, HDF5.H5S_ALL)`.
"""
function get_num_chunks(dataset_id)
    return prod(get_num_chunks_per_dim(dataset_id))
end


"""
    HDF5.get_chunk_length(dataset_id)

Retrieves the chunk size in bytes. Equivalent to `h5d_get_chunk_info(dataset_id, index)[:size]`.
"""
function get_chunk_length(dataset_id)
    type = h5d_get_type(dataset_id)
    chunk = get_chunk(dataset_id)
    return Int(h5t_get_size(type) * prod(chunk))
end

function vlen_get_buf_size(dataset_id)
    type = h5d_get_type(dataset_id)
    space = h5d_get_space(dataset_id)
    h5d_vlen_get_buf_size(dataset_id, type, space)
end

"""
    silence_errors(f::Function)

During execution of the function `f`, disable printing of internal HDF5 library error messages.
"""
function silence_errors(f::Function)
    estack = H5E_DEFAULT
    func, client_data = h5e_get_auto(estack)
    h5e_set_auto(estack, C_NULL, C_NULL)
    try
        return f()
    finally
        h5e_set_auto(estack, func, client_data)
    end
end
