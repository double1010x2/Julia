# Autogenerated wrapper script for Ogg_jll for armv6l-linux-musleabihf
export libogg

JLLWrappers.@generate_wrapper_header("Ogg")
JLLWrappers.@declare_library_product(libogg, "libogg.so.0")
function __init__()
    JLLWrappers.@generate_init_header()
    JLLWrappers.@init_library_product(
        libogg,
        "lib/libogg.so",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@generate_init_footer()
end  # __init__()
