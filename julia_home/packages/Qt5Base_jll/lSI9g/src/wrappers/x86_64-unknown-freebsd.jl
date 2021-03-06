# Autogenerated wrapper script for Qt5Base_jll for x86_64-unknown-freebsd
export libqt5concurrent, libqt5core, libqt5dbus, libqt5gui, libqt5network, libqt5opengl, libqt5printsupport, libqt5sql, libqt5test, libqt5widgets, libqt5xml

using Xorg_libXext_jll
using Xorg_libxcb_jll
using Xorg_xcb_util_wm_jll
using Xorg_xcb_util_image_jll
using Xorg_xcb_util_keysyms_jll
using Xorg_xcb_util_renderutil_jll
using xkbcommon_jll
using Libglvnd_jll
using Fontconfig_jll
using Glib_jll
using Zlib_jll
using CompilerSupportLibraries_jll
using OpenSSL_jll
JLLWrappers.@generate_wrapper_header("Qt5Base")
JLLWrappers.@declare_library_product(libqt5concurrent, "libQt5Concurrent.so.5")
JLLWrappers.@declare_library_product(libqt5core, "libQt5Core.so.5")
JLLWrappers.@declare_library_product(libqt5dbus, "libQt5DBus.so.5")
JLLWrappers.@declare_library_product(libqt5gui, "libQt5Gui.so.5")
JLLWrappers.@declare_library_product(libqt5network, "libQt5Network.so.5")
JLLWrappers.@declare_library_product(libqt5opengl, "libQt5OpenGL.so.5")
JLLWrappers.@declare_library_product(libqt5printsupport, "libQt5PrintSupport.so.5")
JLLWrappers.@declare_library_product(libqt5sql, "libQt5Sql.so.5")
JLLWrappers.@declare_library_product(libqt5test, "libQt5Test.so.5")
JLLWrappers.@declare_library_product(libqt5widgets, "libQt5Widgets.so.5")
JLLWrappers.@declare_library_product(libqt5xml, "libQt5Xml.so.5")
function __init__()
    JLLWrappers.@generate_init_header(Xorg_libXext_jll, Xorg_libxcb_jll, Xorg_xcb_util_wm_jll, Xorg_xcb_util_image_jll, Xorg_xcb_util_keysyms_jll, Xorg_xcb_util_renderutil_jll, xkbcommon_jll, Libglvnd_jll, Fontconfig_jll, Glib_jll, Zlib_jll, CompilerSupportLibraries_jll, OpenSSL_jll)
    JLLWrappers.@init_library_product(
        libqt5concurrent,
        "lib/libQt5Concurrent.so",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@init_library_product(
        libqt5core,
        "lib/libQt5Core.so",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@init_library_product(
        libqt5dbus,
        "lib/libQt5DBus.so",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@init_library_product(
        libqt5gui,
        "lib/libQt5Gui.so",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@init_library_product(
        libqt5network,
        "lib/libQt5Network.so",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@init_library_product(
        libqt5opengl,
        "lib/libQt5OpenGL.so",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@init_library_product(
        libqt5printsupport,
        "lib/libQt5PrintSupport.so",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@init_library_product(
        libqt5sql,
        "lib/libQt5Sql.so",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@init_library_product(
        libqt5test,
        "lib/libQt5Test.so",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@init_library_product(
        libqt5widgets,
        "lib/libQt5Widgets.so",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@init_library_product(
        libqt5xml,
        "lib/libQt5Xml.so",
        RTLD_LAZY | RTLD_DEEPBIND,
    )

    JLLWrappers.@generate_init_footer()
end  # __init__()
