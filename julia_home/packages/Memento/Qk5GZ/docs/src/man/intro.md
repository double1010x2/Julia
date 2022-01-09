# Introduction

## Logging levels

You can globally set the minimum logging level with [`Memento.config!`](@ref).

```julia
julia> Memento.config!("debug")
```
Will log all messages for all loggers at or above "debug".

```julia
julia> Memento.config!("warn")
```
Will only log message at or above the "warn" level.

We can also set the logging level for specific loggers or collections of loggers if we explicitly set the level on an existing logger.

```julia
julia> setlevel!(getlogger("Main"), "info")
```
Will only set the logging level to "info" for the "Main" logger and any future children of the "Main" logger.

By default Memento has 10 logging levels.

Level | Number | Description
--- | --- | ---
not_set | 0 | Will not log anything, but may still propagate messages to its parents.
trace | 5 | Log messages only intended for tracing program execution.
debug | 10 | Log verbose message used for debugging.
info | 20 | Log general information about a program.
notice | 30 | Log important events that are still part of normal execution.
warn | 40 | Log warning that may cause the program to fail.
error | 50 | Log errors and throw or rethrow an error.
critical | 60 | Entire application has crashed.
alert | 70 | The entire application crashed and is not recoverable. Probably need to wake up the sysadmin.
emergency | 80 | System is unusable. Applications shouldn't need to call this so it may be removed in the future.

## Formatting logs

Unless explicitly changed Memento will use a [`DefaultFormatter`](@ref) for handlers.
This [`Formatter`](@ref) takes a format string for mapping log record fields into each log message.
Desired fields are wrapped in curly brackets (ie: `"{msg}"`)

The default format string is `"[{level} | {name}]: {msg}"`, which produces messages that look like

```
[info | root]: my info message.
[warn | root]: my warning message.
...
```

However, you could change this string to just `"{level}: {msg}"`, which would produce messages that look like

```
info: my info message.
warn: my warning message.
...
```

The simplest way to globally change the log format is with [`Memento.config!`](@ref):

```julia
julia> Memento.config!("debug"; fmt="[{level} | {name}]: {msg}")
```

The following fields are available via the [`DefaultRecord`](@ref).

Field | Description
--- | ---
date | The log event date rounded to seconds
level | The log event level as a string
levelnum | The integer value for the log event level
msg | The source log event message
name | The name of the source logger
pid | The pid where the log event occured
threadid | The thread id where the log event occured
lookup | The top StackFrame of the stacktrace for the log event
stacktrace | A StackTrace for the log event

For more details on the [`DefaultFormatter`](@ref) and [`DefaultRecord`](@ref) please see the API docs.
More general information on [`Formatter`s](@ref man_formatters) and [`Record`s](@ref man_records) will be discussed later in this manual.

## Architecture

Memento is a hierarchical logging library, meaning that loggers may have child loggers with their own configurations (e.g., levels, handlers, record types).
By default, logs sent to child loggers propagate up to the root logger, allowing for a simple default configuration at the root.
Memento.jl is designed to separate the responsibilities of logging in libraries (e.g., modules, packages) vs applications.
In general, libraries should focus on logging relevant messages to a library specific child logger, and avoid configuring the child logger directly.

```julia
module MyModule

using Memento

const LOGGER = getlogger(@__MODULE__)

__init__() = Memento.register(LOGGER)
noisy() = debug(LOGGER, "LOUD NOISES!")
hello() = info(LOGGER, "Hello World!")
danger() = warn(LOGGER, "Danger Zone!")

end
```

See the package usage [docs](@ref pkg_usage) for details.

On the other hand, applications are responsible for setting logging levels and attaching custom handlers to specific child loggers as needed.

```julia
using Memento
using MyModule

# Configure global logging level and formatting
Memento.config!("info"; fmt="[{level} | {name}]: {msg}")
push!(getlogger("MyModule"), DefaultHandler("noisy.log"))

# Will only log to noisy.log and not the console
MyModule.noisy()

# These will log to both noisy.log and the console
MyModule.hello()
MyModule.danger()
```

See some of our [configuration recipes](@ref config_recipes) for more details.

There are five main components of Memento.jl that you can manipulate:

1. [Loggers](@ref man_loggers)
2. [Handlers](@ref man_handlers)
3. [Formatters](@ref man_formatters)
4. [Records](@ref man_records)
5. [IO](@ref man_io)

The remainder of this manual will discuss how you can use these components to customize Memento to you particular application.
