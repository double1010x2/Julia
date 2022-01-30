function checkDBGFile(dbg_file::String)
    (length(dbg_file) <= 0) && return nothing
    (isfile(dbg_file)) && (rm(dbg_file))
    return open(dbg_file, "a")
end

function genDBGHeader!(arr_dim, iter::Int, io)
    row, col = arr_dim 
    write(io, "#iter,population")
    for di in range(1, col)
        write(io, ",parameter$di")
    end
    write(io, ",value\n")
end

function saveDBGFile(arr::AbstractArray, val::AbstractArray, iter::Int, io)
    (io == nothing) && return 

    row, col = size(arr)
    (ndims(arr) == 1)   && (row = length(arr))
    (ndims(arr) == 1)   && (col = 1)
    (iter == 1) && (genDBGHeader!((row, col), iter, io))
    for ri in range(1, row)
        write(io, "$iter, $ri")
        for ci in range(1, col)
            write(io, ", $(arr[ri,ci])")
        end
        write(io, ", $(val[ri])\n") 
    end

end
