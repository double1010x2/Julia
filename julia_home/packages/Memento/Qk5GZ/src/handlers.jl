"""
    Handler

Manage formatting `Record`s and logging the resulting `String`. All `Handler` subtypes must
implement at least 1 `log(::Handler, ::Record)` method.

NOTE: Handlers can be useful if you need to special case logging behaviour based on the
`Formatter`, `IO` and/or `Record` types.
"""
abstract type Handler{F<:Formatter} end

"""
    getlevel(::Handler) -> AbstractString

Returns the current handler level.
The default is "not_set".
"""
getlevel(handler::Handler) = "not_set"

"""
    getlevels(::Handler) -> Union{Dict, Nothing}

Get the available log levels for a handler and their associated priorities.
The default is `nothing`, for handlers which do not perform level-based filtering.
"""
getlevels(handler::Handler) = nothing

"""
    getfilters(handler::Handler) -> Array{Memento.Filter}

Returns the filters for the handler.
The default is the standard level-based filter.
"""
getfilters(handler::Handler) = Memento.Filter[]

"""
    log(handler::Handler, rec::Record)

Checks the `Handler` filters and if they all pass then
`emit` the record.
"""
function log(handler::Handler, rec::Record)
    if all(f -> f(rec), getfilters(handler))
        emit(handler, rec)
    end
end

function Memento.Filter(h::Handler)
    function level_filter(rec::Record)
        level = getlevel(rec)
        levels = getlevels(h)

        return levels === nothing || levels[level] >= levels[getlevel(h)]
    end

    Memento.Filter(level_filter)
end

"""
    DefaultHanlder

The DefaultHandler manages any `Formatter`, `IO` and `Record`.

Fields:
- fmt: a `Formatter` for converting `Record`s to `Strings`
- io: an `IO` type for printing `String` to.
- opts: a dictionary of optional arguments such as :is_colorized and :colors
    Ex) ```Dict{Symbol, Any}(
            :is_colorized => true,
            :opts[:colors] => Dict{AbstractString, Symbol}(
                "debug" => :blue,
                "info" => :green,
                ...
            )
        )```
"""
mutable struct DefaultHandler{F, O<:IO} <: Handler{F}
    fmt::F
    io::O
    opts::Dict{Symbol, Any}
    filters::Array{Memento.Filter}
    levels::Union{Dict{AbstractString, Int}, Nothing}
    level::AbstractString
end

"""
    DefaultHandler(
        io::O,
        fmt::F=DefaultFormatter(),
        opts=Dict{Symbol, Any}();
        levels=nothing,
    ) where {F<:Formatter, O<:IO}

Creates a DefaultHandler with the specified IO type.

# Arguments
* `io::IO`: the IO type
* `fmt::Formatter`: the Formatter to use (default to `DefaultFormatter()`)
* `opts::Dict`: the optional arguments (defaults to `Dict{Symbol, Any}()`)
"""
function DefaultHandler(
    io::O,
    fmt::F=DefaultFormatter(),
    opts=Dict{Symbol, Any}();
    levels=nothing,
) where {F<:Formatter, O<:IO}

    setup_opts(opts)
    handler = DefaultHandler(fmt, io, opts, Memento.Filter[], levels, "not_set")
    push!(handler, Memento.Filter(handler))
    return handler
end

"""
    DefaultHandler(
        filename::AbstractString,
        fmt::F=DefaultFormatter(),
        opts=Dict{Symbol, Any}();
        levels=nothing,
    ) where {F<:Formatter}

Creates a DefaultHandler with a IO handle to the specified filename.

# Arguments
* `filename::AbstractString`: the filename of a log file to write to
* `fmt::Formatter`: the Formatter to use (default to `DefaultFormatter()`)
* `opts::Dict`: the optional arguments (defaults to `Dict{Symbol, Any}()`)
"""
function DefaultHandler(
    filename::AbstractString,
    fmt::F=DefaultFormatter(),
    opts=Dict{Symbol, Any}();
    levels=nothing,
) where {F<:Formatter}

    file = open(filename, "a")
    setup_opts(opts)
    handler = DefaultHandler(fmt, file, opts, Memento.Filter[], levels, "not_set")
    push!(handler, Memento.Filter(handler))
    finalizer(h -> close(h.io), handler)
    handler
end

"""
    setup_opts(opts) -> Dict

Sets the default :colors if `opts[:is_colorized] == true`.
"""
function setup_opts(opts)
    if haskey(opts, :colors)
        opts[:is_colorized] = true
    elseif (!haskey(opts, :colors) && haskey(opts, :is_colorized) && opts[:is_colorized])
        # set default colors
        opts[:colors] = Dict{AbstractString, Symbol}(
            "trace" => :normal,
            "debug" => :blue,
            "info" => :green,
            "notice" => :cyan,
            "warn" => :magenta,
            "error" => :red,
            "critical" => :yellow,
            "alert" => :white,
            "emergency" => :black,
        )
    else
        opts[:is_colorized] = false
    end

    opts
end

getfilters(handler::DefaultHandler) = handler.filters

"""
    push!(handler::DefaultHandler, filter::Memento.Filter)

Adds an new `Filter` to the handler.
"""
function Base.push!(handler::DefaultHandler, filter::Memento.Filter)
    push!(handler.filters, filter)
end

getlevels(handler::DefaultHandler) = handler.levels

getlevel(handler::DefaultHandler) = handler.level

"""
    setlevel!(handler::DefaultHandler, level::AbstractString)

Sets the minimum level required to `emit` the record from the handler.
"""
function setlevel!(handler::DefaultHandler, level::AbstractString)
    if handler.levels === nothing
        handler.levels = _log_levels
    end

    handler.levels[level]     # Throw a key error if the levels isn't in levels
    handler.level = level
end

"""
    emit{F, O}(handler::DefaultHandler{F ,O}, rec::Record) where {F<:Formatter, O<:IO}

Handles printing any `Record` with any `Formatter` and `IO` types.
"""
function emit(handler::DefaultHandler{F, O}, rec::Record) where {F<:Formatter, O<:IO}
    level = getlevel(rec)
    str = format(handler.fmt, rec)

    if handler.opts[:is_colorized] && haskey(handler.opts[:colors], level)
        printstyled(
            handler.io,
            string(str,"\n"),
            color=handler.opts[:colors][level],
        )
    else
        println(handler.io, str)
    end

    flush(handler.io)
end

mutable struct Escalator{F} <: Handler{F}
    fmt::F
    filters::Array{Memento.Filter}
    levels::Dict{AbstractString, Int}
    level::AbstractString
end

"""
    Escalator(fmt=DefaultFormatter(); level="warn", levels=nothing)

Escalates any logs it sees above a certain `level` and throws an `EscalationError`.

# Arguments

- `fmt::Formatter`: for converting `Record`s to error messages `Strings`

# Keyword Arguments
- `level`: threshold level for when to error, otherwise this is a no-op
- `levels`: an alternate levels dictionary if we're considering non-default levels
"""
function Escalator(
    fmt::F=DefaultFormatter(); level="warn", levels=nothing
) where {F<:Formatter}
    handler = Escalator(
        fmt,
        Memento.Filter[],
        levels === nothing ? Memento._log_levels : levels,
        level,
    )

    push!(handler, Memento.Filter(handler))
    return handler
end

getfilters(handler::Escalator) = handler.filters
Base.push!(handler::Escalator, filter::Memento.Filter) = push!(handler.filters, filter)
getlevels(handler::Escalator) = handler.levels
getlevel(handler::Escalator) = handler.level
function setlevel!(handler::Escalator, level::AbstractString)
    handler.levels[level]     # Throw a key error if the levels isn't in levels
    handler.level = level
end

emit(handler::Escalator, rec::Record) = throw(EscalationError(format(handler.fmt, rec)))
