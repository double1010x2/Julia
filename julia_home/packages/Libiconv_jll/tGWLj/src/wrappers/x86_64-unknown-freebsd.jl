# Autogenerated wrapper script for Libiconv_jll for x86_64-unknown-freebsd
export libcharset, libiconv

JLLWrappers.@generate_wrapper_header("Libiconv")
JLLWrappers.@declare_library_product(libcharset, "libcharset.so.1")
JLLWrappers.@declare_library_product(libiconv, "libiconv.so.2")
function __init__()
    JLLWrappers.@generate_init_header()
    JLLWrappers.@init_library_product(
        libcharset,
        "lib/libcharset.so",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@init_library_product(
        libiconv,
        "lib/libiconv.so",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@generate_init_footer()
end  # __init__()
