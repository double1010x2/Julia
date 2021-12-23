# Test suite

GPUArrays provides an extensive test suite that covers all of the functionality that should
be available after implementing the required interfaces. This test suite is part of this
package, but for dependency reasons it is not available when importing the package. Instead,
you should include the code from your `runtests.jl` as follows:

```julia
import GPUArrays
gpuarrays = pathof(GPUArrays)
gpuarrays_root = dirname(dirname(gpuarrays))
include(joinpath(gpuarrays_root, "test", "testsuite.jl"))
```

This however implies that the test system will not know about extra dependencies that are
required by the test suite. To remedy this, you should add the following dependencies to
your `Project.toml`:

```
[extras]
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
...

[targets]
test = [..., "FFTW", "ForwardDiff", "FillArrays"]
```

With this set-up, you can run the test suite like this:

```julia
TestSuite.test(MyGPUArrayType)
```
If you don't want to run the whole suite, you can also run parts of it:


```julia
T = JLArray
GPUArrays.allowscalar(false) # fail tests when slow indexing path into Array type is used.

TestSuite.test_gpuinterface(T) # interface functions like gpu_call, threadidx, etc
TestSuite.test_base(T) # basic functionality like launching a kernel on the GPU and Base operations
TestSuite.test_blas(T) # tests the blas interface
TestSuite.test_broadcasting(T) # tests the broadcasting implementation
TestSuite.test_construction(T) # tests all kinds of different ways of constructing the array
TestSuite.test_linalg(T) # linalg function tests
TestSuite.test_mapreduce(T) # mapreduce sum, etc
TestSuite.test_indexing(T) # indexing tests
TestSuite.test_random(T) # randomly constructed arrays
TestSuite.test_io(T)
```
