# Autogenerated wrapper script for PCRE_jll for armv7l-linux-gnueabihf-cxx03
export libpcre

JLLWrappers.@generate_wrapper_header("PCRE")
JLLWrappers.@declare_library_product(libpcre, "libpcre.so.1")
function __init__()
    JLLWrappers.@generate_init_header()
    JLLWrappers.@init_library_product(
        libpcre,
        "lib/libpcre.so",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@generate_init_footer()
end  # __init__()
