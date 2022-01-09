.. _dataviews:

Data Views
=============

There exist a wide variety of machine learning algorithms, each
of which unique in their own way. Yet, there are clear
similarities concerning how these algorithms utilize the data set
during model training. In fact, most algorithms belong to at
least one of the following categories.

- Some algorithms use the whole training data in every iteration
  of the training procedure. If that is the case, then it is not
  necessary to partition or otherwise prepare the data any
  further.

- In contrast to this, many of the modern algorithms prefer to
  process the training data piece by piece using smaller chunks.
  These "chunks" are usually of some fixed size and are commonly
  referred to as *mini batches*.

- Yet another (but overlapping) group of algorithms processes the
  training data just one single observation at a time.

What we can learn from this is that regardless of the concrete
algorithm and all its details, it is quite likely that at some
point during a machine learning experiment, we need to iterate
over the training data in a particular way. Most of the time we
either iterate over the training data one observation at a time,
or one mini-batch at at time.

This package provides multiple approaches to accomplish such a
functionality. This "multiple" is necessary, because there are
different forms of data sources, that have very different
properties. Recall how we differentiated between data container
and data iterators. In this document, however, we will solely
focus on data sources that are considered :ref:`container`.

As Vector of Observations
---------------------------

One could be inclined to believe, that in order to iterate over
some data container one observation at a time, it should suffice
to just iterate over the data container itself. While this may
work as expected for some data container types, it will not work
in general.

Consider the following two example data containers, the matrix
``X`` and the vector ``y``. While both are different types of
data container, we will make each one store exactly 5
observations.

.. code-block:: jlcon

   julia> X = rand(2, 5)
   2×5 Array{Float64,2}:
    0.226582  0.933372  0.505208   0.0443222  0.812814
    0.504629  0.522172  0.0997825  0.722906   0.245457

   julia> y = rand(5)
   5-element Array{Float64,1}:
    0.11202
    0.000341996
    0.380001
    0.505277
    0.841177

When we iterate over ``y``, it turns out we also iterate over
each observation. So in other words, for a ``Vector`` it would
work to just iterate over the data container itself, if our goal
is to process it one observation at a time.

.. code-block:: jlcon

   julia> foreach(println, y)
   0.11201971329803984
   0.0003419958128361156
   0.3800005018641015
   0.5052774551949404
   0.8411766784932724

On the other hand, when we iterate over ``X``, we iterate over
all individual array elements, and *not* over what we consider to
be the observations. In general that is the behaviour we want for
a ``Array``, but it is not in line with our domain interpretation
of a data container.

.. code-block:: jlcon

   julia> foreach(println, X)
   0.22658190197881312
   0.5046291972412908
   0.9333724636943255
   0.5221721267193593
   0.5052080505550971
   0.09978246027514359
   0.04432218813798294
   0.7229058081423172
   0.8128138585478044
   0.24545709827626805

This means, that we need a more general approach for iterating
over a data container one observation at a time. Recall how data
containers have the nice property of knowing how many
observations they contain, and how to access each individual
observation. Because of this we need not even limit ourselves to
just iteration here, instead we could just create a new "view".
For that purpose we provide the type :class:`ObsView`, which can
be used to treat any data container as a vector-like
representation of that data container, where each vector element
corresponds to a single observation.

.. class:: ObsView <: DataView <: AbstractVector

   Lazy representation of a data container as a vector of
   individual observations.

   Any data access is delayed until ``getindex`` is called, and
   even ``getindex`` returns the result of :func:`datasubset`
   which in general avoids data movement until :func:`getobs` is
   invoked.

.. function:: obsview(data, [obsdim]) -> ObsView

   Create a :class:`ObsView` for the given `data` container. It
   will serve as a vector-like view into `data`, where every
   element of the vector points to a single observation in
   `data`.

   :param data: The object representing a data container.

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        type-stable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

Let us consider our toy matrix ``X`` again, which we will
interpret as containing 5 observations with 2 features each. This
time we pass it to :func:`obsview` before iterating over it.
Notice how the resulting :class:`ObsView` will look like a vector
of vectors.  As we will see from the type, each element of the
:class:`ObsView` ``ov`` is just a ``SubArray`` into ``X``. As
such, no data from ``X`` is copied.

.. code-block:: jlcon

   julia> X = rand(2, 5)
   2×5 Array{Float64,2}:
    0.226582  0.933372  0.505208   0.0443222  0.812814
    0.504629  0.522172  0.0997825  0.722906   0.245457

   julia> ov = obsview(X)
   5-element obsview(::Array{Float64,2}, ObsDim.Last()) with element type SubArray{Float64,1,Array{Float64,2},Tuple{Colon,Int64},true}:
    [0.226582,0.504629]
    [0.933372,0.522172]
    [0.505208,0.0997825]
    [0.0443222,0.722906]
    [0.812814,0.245457]

   julia> ov[2] # access second observation
   2-element SubArray{Float64,1,Array{Float64,2},Tuple{Colon,Int64},true}:
    0.933372
    0.522172

   julia> foreach(println, ov) # now we iterate over observation
   [0.226582,0.504629]
   [0.933372,0.522172]
   [0.505208,0.0997825]
   [0.0443222,0.722906]
   [0.812814,0.245457]

If there is more than one array dimension, all but the
observation dimension are implicitly assumed to be features (i.e.
part of that observation). As we have seen with ``X`` in the
example above, the default assumption is that the last array
dimension enumerates the observations. This can be overwritten by
explicitly specifying the ``obsdim``. In the following code
snippet we treat ``X`` as a data set that has 2 observations with
5 features each.

.. code-block:: jlcon

   julia> X = rand(2, 5)
   2×5 Array{Float64,2}:
    0.226582  0.933372  0.505208   0.0443222  0.812814
    0.504629  0.522172  0.0997825  0.722906   0.245457

   julia> ov = obsview(X, obsdim = 1)
   2-element obsview(::Array{Float64,2}, ObsDim.Constant{1}()) with element type SubArray{Float64,1,Array{Float64,2},Tuple{Int64,Colon},true}:
    [0.226582,0.933372,0.505208,0.0443222,0.812814]
    [0.504629,0.522172,0.0997825,0.722906,0.245457]

   julia> ov = obsview(X, ObsDim.First()); # same as above but type-stable

Similarly, we can also call :func:`obsview` with our toy vector
``y``. Recall that a ``Vector`` is just an ``Array`` with only
one dimension. This example will help demonstrate how an
:class:`ObsView` handles data container that are already in a
vector-like form.

.. code-block:: jlcon

   julia> y = rand(5)
   5-element Array{Float64,1}:
    0.11202
    0.000341996
    0.380001
    0.505277
    0.841177

   julia> ov = obsview(y)
   5-element obsview(::Array{Float64,1}, ObsDim.Last()) with element type SubArray{Float64,0,Array{Float64,1},Tuple{Int64},false}:
    0.11202
    0.000341996
    0.380001
    0.505277
    0.841177

   julia> ov[2] # access second observation
   0-dimensional SubArray{Float64,0,Array{Float64,1},Tuple{Int64},false}:
   0.000341996

On first glance, the result of indexing into ``ov`` may seem
unintuitive. Why does it return a :math:`0`-dimensional
``SubArray`` instead of simply the value? The main reason for
this behaviour is that we try to avoid data movement unless
:func:`getobs` is called. Until that point, we just create
subsets into the original data container.

.. code-block:: jlcon

   julia> getobs(ov[2])
   0.0003419958128361156

   julia> getobs(ov)
   5-element Array{Float64,1}:
    0.11202
    0.000341996
    0.380001
    0.505277
    0.841177

   julia> getobs(ov, 2)
   0.0003419958128361156

You may have noted in all the examples so far, that creating an
:class:`ObsView` preserves the order of the observations. This is
of course on purpose and the desired behaviour. However, since
:class:`ObsView` is commonly used as an iterator, one may be
inclined to prefer iterating over the data in a random order. To
do so, simply combine the functions :func:`obsview` and
:func:`shuffleobs`.

.. code-block:: jlcon

   julia> ov = obsview(shuffleobs(y))
   5-element obsview(::SubArray{Float64,1,Array{Float64,1},Tuple{Array{Int64,1}},false}, ObsDim.Last()) with element type SubArray{Float64,0,Array{Float64,1},Tuple{Int64},false}:
    0.505277
    0.11202
    0.841177
    0.380001
    0.000341996

   julia> ov = shuffleobs(obsview(y)) # also possible
   5-element obsview(::SubArray{Float64,1,Array{Float64,1},Tuple{Array{Int64,1}},false}, ObsDim.Last()) with element type SubArray{Float64,0,Array{Float64,1},Tuple{Int64},false}:
    0.505277
    0.380001
    0.000341996
    0.841177
    0.11202

It is also possible to link multiple different data containers
together on an per-observation level. To do that, simply put all
the relevant data container into a single ``Tuple``, before
passing it to :func:`obsview`. Each element of the resulting
:class:`ObsView` will then be a ``Tuple``, with the resulting
observation in the same tuple position.

.. code-block:: jlcon

   julia> ov = obsview((X, y))
   5-element obsview(::Tuple{Array{Float64,2},Array{Float64,1}}, (ObsDim.Last(),ObsDim.Last())) with element type Tuple{SubArray{Float64,1,Array{Float64,2},Tuple{Colon,Int64},true},SubArray{Float64,0,Array{Float64,1},Tuple{Int64},false}}:
    ([0.226582,0.504629],0.11202)
    ([0.933372,0.522172],0.000341996)
    ([0.505208,0.0997825],0.380001)
    ([0.0443222,0.722906],0.505277)
    ([0.812814,0.245457],0.841177)

   julia> ov[2] # access second observation
   ([0.933372,0.522172],0.000341996)

   julia> typeof(ov[2])
   Tuple{SubArray{Float64,1,Array{Float64,2},Tuple{Colon,Int64},true},SubArray{Float64,0,Array{Float64,1},Tuple{Int64},false}}

It is worth pointing out, that the tuple elements (i.e. data
container), that are passed to :func:`obsview`, need not be of
the same type, nor of the same shape. You can observe this in the
code above, where ``X`` is a ``Matrix`` while ``y`` is a
``Vector``. Note, however, that all tuple elements must be data
containers themselves. Furthermore, they all must contain the
same exact number of observations.

As Vector of Batches
-----------------------

Another common use case is to iterate over the given data set in
small equal-sized chunks. These chunks are usually referred to as
*mini-batches*.

Not unlike :class:`ObsView`, this package provides a vector-like
type called :class:`BatchView`, that can be used to treat any
data container as a vector of equal-sized batches.

.. class:: BatchView <: DataView <: AbstractVector

   Lazy representation of a data container as a vector of
   batches. Each batch will contain an equal amount of
   observations in them. In the case that the number ob
   observations is not dividable by the specified (or inferred)
   batch-size, the remaining observations will be ignored.

   Any data access is delayed until ``getindex`` is called, and
   even ``getindex`` returns the result of :func:`datasubset`
   which in general avoids data movement until :func:`getobs` is
   invoked.

.. function:: batchview(data, [size|maxsize], [count], [obsdim]) -> BatchView

   Create a :class:`BatchView` for the given `data` container. It
   will serve as a vector-like view into `data`, where every
   element of the vector points to a batch of `size` observations
   from `data`. The number of batches and the batch-size can be
   specified using (keyword) parameters `count` and `size` (or
   alternatively `maxsize`).

   In the case that the size of the dataset is not dividable by
   the specified (or inferred) `size`, the remaining observations
   will be ignored. If `maxsize` is provided instead of `size`,
   then the next dividable size will be used such that no
   observations are ignored.

   :param data: The object representing a data container.

   :param Integer size: Optional. The exact number of observations
                        in each batch.

   :param Integer maxsize: \
        Optional alternative to `size`. The maximal number of
        observations in each batch, such that all observations
        get used.

   :param Integer count: \
        Optional. The number of batches that should be used. This
        will also we the length of the return value.

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        type-stable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

Consider the following toy data-matrix ``X``, which we will
interpret as containing a total of 5 observations, where each
observation consists of 2 features.

.. code-block:: jlcon

   julia> X = rand(2, 5)
   2×5 Array{Float64,2}:
    0.226582  0.933372  0.505208   0.0443222  0.812814
    0.504629  0.522172  0.0997825  0.722906   0.245457

Using a prime number for the total number of observations makes
this data container a particularly interesting example for using
:func:`batchview`. Unless we choose a batch-size of ``1`` or
``5``, there is no way to iterate the whole data in terms of
equally-sized batches. :class:`BatchView` deals with such edge
cases by ignoring the excess observations with an informative
message.

.. code-block:: jlcon

   julia> bv = batchview(X, size = 2)
   INFO: The specified values for size and/or count will result in 1 unused data points
   2-element batchview(::Array{Float64,2}, 2, 2, ObsDim.Last()) with element type SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}:
    [0.226582 0.933372; 0.504629 0.522172]
    [0.505208 0.0443222; 0.0997825 0.722906]

   julia> bv = batchview(X, maxsize = 2)
   5-element batchview(::Array{Float64,2}, 1, 5, ObsDim.Last()) with element type SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}:
    [0.226582; 0.504629]
    [0.933372; 0.522172]
    [0.505208; 0.0997825]
    [0.0443222; 0.722906]
    [0.812814; 0.245457]

You can query the size of each batch by using the function
:func:`batchsize` on any :class:`BatchView`.

.. code-block:: jlcon

   julia> batchsize(bv)
   2

Similar to :class:`ObsView`, a :class:`BatchView` acts like a
vector and can be used as such. The one big difference to the
former is that each element is now a batch of ``X`` instead of a
single observation.

.. code-block:: jlcon

   julia> bv[2] # access second batch
   2×2 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}:
    0.505208   0.0443222
    0.0997825  0.722906

Naturally, :func:`batchview` also supports the optional parameter
``obsdim``, which can be used to specify which dimension denotes
the observation. If that concept of dimensionality does not make
sense for the given data container, then ``obsdim`` can simply be
omitted.

.. code-block:: jlcon

   julia> bv = batchview(X', size = 2, obsdim = 1) # note the transpose
   INFO: The specified values for size and/or count will result in 1 unused data points
   2-element batchview(::Array{Float64,2}, 2, 2, ObsDim.Constant{1}()) with element type SubArray{Float64,2,Array{Float64,2},Tuple{UnitRange{Int64},Colon},false}:
    [0.226582 0.504629; 0.933372 0.522172]
    [0.505208 0.0997825; 0.0443222 0.722906]

So far we used the parameter ``size`` to explicitly specify how
many observation we want to be in each batch. Alternatively, we
can also use the parameter ``count`` to specify the total number
of batches that we would like to use.

.. code-block:: jlcon

   julia> bv = batchview(X, count = 4)
   INFO: The specified values for size and/or count will result in 1 unused data points
   4-element batchview(::Array{Float64,2}, 1, 4, ObsDim.Last()) with element type SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}:
    [0.226582; 0.504629]
    [0.933372; 0.522172]
    [0.505208; 0.0997825]
    [0.0443222; 0.722906]

   julia> bv[2] # access second batch
   2×1 SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true}:
    0.933372
    0.522172

Note how in the above example, the inferred batch-size is ``1``.
Arguably, this makes the resulting :class:`BatchView`, ``bv``,
appear very similar to an :class:`ObsView`. The big difference,
though, is that :class:`BatchView` preserves the shape on
indexing. Consequently, each element of ``bv`` is a subtype of
``AbstractMatrix`` and not ``AbstractVector``.

It is also possible to call :func:`batchview` with multiple data
containers wrapped in a ``Tuple``. Note, however, that all data
containers must have the same total number of observations. Using
a tuple this way will link those data containers together on a
per-observation basis.

.. code-block:: jlcon

   julia> y = rand(5)
   5-element Array{Float64,1}:
    0.11202
    0.000341996
    0.380001
    0.505277
    0.841177

   julia> bv = batchview((X, y))
   INFO: The specified values for size and/or count will result in 1 unused data points
   2-element batchview(::Tuple{Array{Float64,2},Array{Float64,1}}, 2, 2, (ObsDim.Last(),ObsDim.Last())) with element type Tuple{SubArray{Float64,2,Array{Float64,2},Tuple{Colon,UnitRange{Int64}},true},SubArray{Float64,1,Array{Float64,1},Tuple{UnitRange{Int64}},true}}:
   ([0.226582 0.933372; 0.504629 0.522172], [0.11202,0.000341996])
   ([0.505208 0.0443222; 0.0997825 0.722906], [0.380001,0.505277])

   julia> bv[2]
   ([0.505208 0.0443222; 0.0997825 0.722906], [0.380001,0.505277])

As Vector of Sequences
-----------------------

Time series data, or more generally *sequence data*, often
requires a special type of preparation in order to work with it
in a machine learning experiment. The big difference to "normal"
data is that in sequence data the observations are not
independent from each other. For example if you think of a piece
of text as a sequence of words (i.e. each word is an
observation), you'll notice that there is an inherent order in
the data.

.. code-block:: jlcon

   julia> data = split("The quick brown fox jumps over the lazy dog")
   9-element Array{SubString{String},1}:
    "The"
    "quick"
    "brown"
    "fox"
    "jumps"
    "over"
    "the"
    "lazy"
    "dog"

Unlabeled Windows
~~~~~~~~~~~~~~~~~~~~~~~

There are some common scenarios when working with sequence data
such as the above. Before we think about more practical use cases
that require labeled windows, let us quickly consider the case
that we would like to process our sequence data in chunks of
equally sized windows.

.. function:: slidingwindow(data, size, [stride], [obsdim])

   Return a vector-like view of the `data` for which each element
   is a fixed size "window" of `size` adjacent observations. By
   default these windows are not overlapping.

   :param data: The object representing a data container.

   :param Integer size: The number of observations in each window.

   :param Integer stride: \
        Optional. The step size between the starting observation
        of susequent windows. Defaults to `size`.

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        type-stable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

Its worth pointing out that only complete windows are included in
the output. This implies that it is possible for excess
observations to be omitted from the view. The following code
snippet shows an example that partitions 22 observation into 4
windows, where the last two observations are omitted.

.. code-block:: jlcon

   julia> A = slidingwindow(1:22, 4)
   5-element slidingwindow(::UnitRange{Int64}, 4) with element type SubArray{Int64,1,UnitRange{Int64},Tuple{UnitRange{Int64}},true}:
    [1, 2, 3, 4]
    [5, 6, 7, 8]
    [9, 10, 11, 12]
    [13, 14, 15, 16]
    [17, 18, 19, 20]

Note that the values of the given data are not actually copied.
Instead the function :func:`datasubset` is called when
``getindex`` is invoked. To actually get a copy of the data at
some window use the function :func:`getobs`.

.. code-block:: jlcon

   julia> A[2]
   4-element SubArray{Int64,1,UnitRange{Int64},Tuple{UnitRange{Int64}},true}:
    5
    6
    7
    8

   julia> getobs(A, 2)
   4-element Array{Int64,1}:
    5
    6
    7
    8

Up to this point the behaviour may be very reminiscent of a
:func:`batchview`, but this is where the similarities end. The
optional parameter ``stride`` can be used to specify the distance
between the start elements of each adjacent window. By default
the stride is equal to the window size.

.. code-block:: jlcon

   julia> A = slidingwindow(1:22, 4, stride=2)
   10-element slidingwindow(::UnitRange{Int64}, 4, stride = 2) with element type SubArray{Int64,1}:
    [1, 2, 3, 4]
    [3, 4, 5, 6]
    [5, 6, 7, 8]
    [7, 8, 9, 10]
    [9, 10, 11, 12]
    [11, 12, 13, 14]
    [13, 14, 15, 16]
    [15, 16, 17, 18]
    [17, 18, 19, 20]
    [19, 20, 21, 22]

   julia> A = slidingwindow(data, 4, stride=2)
   3-element slidingwindow(::Array{SubString{String},1}, 4, stride = 2) with element type SubArray{...}:
    ["The", "quick", "brown", "fox"]
    ["brown", "fox", "jumps", "over"]
    ["jumps", "over", "the", "lazy"]

.. _sequences:

Labeled Windows
~~~~~~~~~~~~~~~~~~~~~~~

Now that we have seen the general idea of ``slidingwindow``, let
us consider a more practical variation of it. A conceptually
simple use case may be that we want to predict the next word in a
sentence given all the words that came before it (e.g. for
autocompletion).

An interesting aspect of sequence prediction is that we can
transform an unlabeled sequence into a number of labeled
sub-sequences. Let's again use our original (unlabeled) data for
this.

.. code-block:: jlcon

   julia> data = split("The quick brown fox jumps over the lazy dog")
   9-element Array{SubString{String},1}:
    "The"
    "quick"
    "brown"
    "fox"
    "jumps"
    "over"
    "the"
    "lazy"
    "dog"

If we were to train a model that given two words should predict
the next word, we would need to rearrange our ``data`` quite a
bit. To make this process more convenient we provide a custom
method for ``slidingwindow`` that expects an target-index
function ``f`` as first parameter.

.. function:: slidingwindow(f, data, size, [stride], [excludetarget], [obsdim])

   Return a vector-like view of the `data` for which each element
   is a tuple of two elements:

   1. A fixed size "window" of `size` adjacent observations. By
      default these windows are not overlapping. This can be
      changed by explicitly specifying a `stride`.

   2. A single target (or vector of targets) for the window. The
      content of the target(s) is defined by the label-index
      function `f`.

   :param Function f: \
        A unary function that takes the index of
        the first observation in the current window and should
        return the index (or indices) of the associated target(s)
        for that window.

   :param data: The object representing a data container.

   :param Integer size: The number of observations in each window.

   :param Integer stride: \
        Optional. The step size between the starting observation
        of susequent windows. Defaults to `size`.

   :param Bool excludetarget: \
        Should a target index returned by `f` also occur in the
        window, then setting this to ``true`` will make sure that
        such elements are removed from the window. Defaults to
        ``false``.

   :param obsdim: \
        Optional. If it makes sense for the type of `data`, then
        `obsdim` can be used to specify which dimension of `data`
        denotes the observations. It can be specified in a
        type-stable manner as a positional argument, or as a more
        convenient keyword parameter. See :ref:`obsdim` for more
        information.

Note that only complete and in-bound windows are included in the
output, which implies that it is possible for excess observations
to be omitted from the resulting view.

.. code-block:: jlcon

   julia> A = slidingwindow(i->i+2, data, 2, stride=1)
   7-element slidingwindow(::##9#10, ::Array{SubString{String},1}, 2, stride = 1) with element type Tuple{...}:
    (["The", "quick"], "brown")
    (["quick", "brown"], "fox")
    (["brown", "fox"], "jumps")
    (["fox", "jumps"], "over")
    (["jumps", "over"], "the")
    (["over", "the"], "lazy")
    (["the", "lazy"], "dog")

   julia> A = slidingwindow(i->i-1, data, 2, stride=1)
   7-element slidingwindow(::##11#12, ::Array{SubString{String},1}, 2, stride = 1) with element type Tuple{...}:
    (["quick", "brown"], "The")
    (["brown", "fox"], "quick")
    (["fox", "jumps"], "brown")
    (["jumps", "over"], "fox")
    (["over", "the"], "jumps")
    (["the", "lazy"], "over")
    (["lazy", "dog"], "the")

As hinted above, it is also allowed for ``f`` to return a vector of
indices. This can be useful for emulating techniques such as
skip-gram.

.. code-block:: jlcon

   julia> A = slidingwindow(i->[i-2:i-1; i+1:i+2], data, 1)
   5-element slidingwindow(::##11#12, ::Array{SubString{String},1}, 1) with element type Tuple{...}:
    (["brown"], ["The", "quick", "fox", "jumps"])
    (["fox"], ["quick", "brown", "jumps", "over"])
    (["jumps"], ["brown", "fox", "over", "the"])
    (["over"], ["fox", "jumps", "the", "lazy"])
    (["the"], ["jumps", "over", "lazy", "dog"])

Should it so happen that the targets overlap with the features,
then the affected observation(s) will be present in both. To
change this behaviour one can set the optional parameter
``excludetarget = true``. This will remove the target(s) from the
feature window.

.. code-block:: jlcon

   julia> slidingwindow(i->i+2, data, 5, stride = 1, excludetarget = true)
   5-element slidingwindow(::##17#18, ::Array{SubString{String},1}, 5, stride = 1) with element type Tuple{...}:
    (["The", "quick", "fox", "jumps"], "brown")
    (["quick", "brown", "jumps", "over"], "fox")
    (["brown", "fox", "over", "the"], "jumps")
    (["fox", "jumps", "the", "lazy"], "over")
    (["jumps", "over", "lazy", "dog"], "the")
