# Autogenerated wrapper script for x265_jll for armv7l-linux-musleabihf
export libx265, x265

JLLWrappers.@generate_wrapper_header("x265")
JLLWrappers.@declare_library_product(libx265, "libx265.so.199")
JLLWrappers.@declare_executable_product(x265)
function __init__()
    JLLWrappers.@generate_init_header()
    JLLWrappers.@init_library_product(
        libx265,
        "lib/libx265.so",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@init_executable_product(
        x265,
        "bin/x265",
    )

    JLLWrappers.@generate_init_footer()
end  # __init__()
