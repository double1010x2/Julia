# Autogenerated wrapper script for Xorg_libXfixes_jll for x86_64-linux-musl
export libXfixes

using Xorg_libX11_jll
JLLWrappers.@generate_wrapper_header("Xorg_libXfixes")
JLLWrappers.@declare_library_product(libXfixes, "libXfixes.so.3")
function __init__()
    JLLWrappers.@generate_init_header(Xorg_libX11_jll)
    JLLWrappers.@init_library_product(
        libXfixes,
        "lib/libXfixes.so",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@generate_init_footer()
end  # __init__()