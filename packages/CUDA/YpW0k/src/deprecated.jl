# Deprecated functionality

@deprecate CuDevice(ctx::CuContext) device(ctx)
@deprecate CuCurrentDevice() current_device()
@deprecate CuCurrentContext() current_context()
@deprecate CuContext(ptr::Union{Ptr,CuPtr}) context(ptr)
@deprecate CuDevice(ptr::Union{Ptr,CuPtr}) device(ptr)
