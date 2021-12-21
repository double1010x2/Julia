# Autogenerated wrapper script for x265_jll for x86_64-w64-mingw32
export libx265, x265

JLLWrappers.@generate_wrapper_header("x265")
JLLWrappers.@declare_library_product(libx265, "libx265.dll")
JLLWrappers.@declare_executable_product(x265)
function __init__()
    JLLWrappers.@generate_init_header()
    JLLWrappers.@init_library_product(
        libx265,
        "bin\\libx265.dll",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@init_executable_product(
        x265,
        "bin\\x265.exe",
    )

    JLLWrappers.@generate_init_footer()
end  # __init__()