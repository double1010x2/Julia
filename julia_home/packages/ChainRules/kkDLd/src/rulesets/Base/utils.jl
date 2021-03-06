# imag(conj(x) * y) avoiding computing the real part if possible
@inline _imagconjtimes(x, y) = imag(conj(x) * y)
@inline function _imagconjtimes(x::Complex, y::Complex)
    return muladd(-imag(x), real(y), real(x) * imag(y))
end
@inline _imagconjtimes(x::Real, y::Complex) = x * imag(y)
@inline _imagconjtimes(x::Complex, y::Real) = -imag(x) * y
@inline _imagconjtimes(x::Real, y::Real) = ZeroTangent()
