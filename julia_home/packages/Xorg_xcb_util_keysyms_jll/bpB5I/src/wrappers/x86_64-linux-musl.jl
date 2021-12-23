# Autogenerated wrapper script for Xorg_xcb_util_keysyms_jll for x86_64-linux-musl
export libxcb_keysyms

using Xorg_xcb_util_jll
JLLWrappers.@generate_wrapper_header("Xorg_xcb_util_keysyms")
JLLWrappers.@declare_library_product(libxcb_keysyms, "libxcb-keysyms.so.1")
function __init__()
    JLLWrappers.@generate_init_header(Xorg_xcb_util_jll)
    JLLWrappers.@init_library_product(
        libxcb_keysyms,
        "lib/libxcb-keysyms.so",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@generate_init_footer()
end  # __init__()
