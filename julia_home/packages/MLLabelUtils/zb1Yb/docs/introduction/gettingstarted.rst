Getting Started
================

MLLabelUtils is the result of a collaborative effort to design an
efficient but also convenient-to-use library for working with
the most commonly utilized class-label encodings in Machine Learning.
As such, this package provides functionality to derive or assert
properties about some label-encoding or target array, as well as
the functions needed to convert given targets into a different
format.

Installation
-------------

To install `MLLabelUtils.jl
<https://github.com/JuliaML/MLLabelUtils.jl>`_, start up `Julia
<http://julialang.org/>`_ and type the following code-snipped
into the REPL. It makes use of the native Julia package manger.

.. code-block:: julia

    Pkg.add("MLLabelUtils")

Additionally, for example if you encounter any sudden issues,
or in the case you would like to contribute to the package,
you can manually choose to be on the latest (untagged) version.

.. code-block:: julia

    Pkg.checkout("MLLabelUtils")

Overview
------------

Let us take a look at some examples (with only minor explanation)
to get a feeling for what one can do with this package. Once
installed the package can be imported just as any other Julia
package.

.. code-block:: julia

   using MLLabelUtils

For starters, the library provides a few utility functions to
compute various properties of the target array. These include the
number of labels (see :func:`nlabel`), the labels themselves (see
:func:`label`), and a mapping from label to the elements of the
target array (see :func:`labelmap` and :func:`labelfreq`).

.. code-block:: jlcon

   julia> true_targets = [0, 1, 1, 0, 0];

   julia> label(true_targets)
   2-element Array{Int64,1}:
    1
    0

   julia> nlabel(true_targets)
   2

   julia> labelmap(true_targets)
   Dict{Int64,Array{Int64,1}} with 2 entries:
     0 => [1,4,5]
     1 => [2,3]

   julia> labelfreq(true_targets)
   Dict{Int64,Int64} with 2 entries:
     0 => 3
     1 => 2

.. tip::

   Because :func:`labelfreq` utilizes a ``Dict`` to store its result,
   it is straight forward to visualize the class distribution
   (using the absolute frequencies) right in the REPL using the
   `UnicodePlots.jl <https://github.com/Evizero/UnicodePlots.jl>`_
   package.

   .. code-block:: jlcon

      julia> using UnicodePlots
      julia> barplot(labelfreq([:yes,:no,:no,:maybe,:yes,:yes]), symb="#")
      #        ┌────────────────────────────────────────┐
      #    yes │##################################### 3 │
      #  maybe │############ 1                          │
      #     no │######################### 2             │
      #        └────────────────────────────────────────┘

If you find yourself writing some custom function that is intended
to train some specific supervised model, chances are that you want to
assert if the given targets are in the correct encoding that the model
requires. We provide a few functions for such a scenario, namely
:func:`labelenc` and :func:`islabelenc`.

.. code-block:: jlcon

   julia> true_targets = [0, 1, 1, 0, 0];

   julia> labelenc(true_targets) # determine encoding using heuristics
   MLLabelUtils.LabelEnc.ZeroOne{Int64,Float64}(0.5)

   julia> islabelenc(true_targets, LabelEnc.ZeroOne)
   true

   julia> islabelenc(true_targets, LabelEnc.ZeroOne(Int))
   true

   julia> islabelenc(true_targets, LabelEnc.ZeroOne(Float32))
   false

   julia> islabelenc(true_targets, LabelEnc.MarginBased)
   false

In the case that it turns out the given targets are in the wrong
encoding you may want to convert them into the format you require.
For that purpose we expose the function :func:`convertlabel`.

.. code-block:: jlcon

   julia> true_targets = [0, 1, 1, 0, 0];

   julia> convertlabel(LabelEnc.MarginBased, true_targets)
   5-element Array{Int64,1}:
    -1
     1
     1
    -1
    -1

   julia> convertlabel(LabelEnc.MarginBased(Float64), true_targets)
   5-element Array{Float64,1}:
    -1.0
     1.0
     1.0
    -1.0
    -1.0

   julia> convertlabel([:yes,:no], true_targets)
   5-element Array{Symbol,1}:
    :no
    :yes
    :yes
    :no
    :no

   julia> convertlabel(LabelEnc.OneOfK, true_targets)
   2×5 Array{Int64,2}:
    0  1  1  0  0
    1  0  0  1  1

   julia> convertlabel(LabelEnc.OneOfK{Bool}, true_targets)
   2×5 Array{Bool,2}:
    false   true   true  false  false
     true  false  false   true   true

   julia> convertlabel(LabelEnc.OneOfK{Float64}, true_targets, obsdim=1)
   5×2 Array{Float64,2}:
    0.0  1.0
    1.0  0.0
    1.0  0.0
    0.0  1.0
    0.0  1.0

It may be interesting to point out explicitly that we provide
:class:`LabelEnc.OneVsRest` to conveniently convert a multi-class
problem into a two-class problem.

.. code-block:: jlcon

   julia> convertlabel(LabelEnc.OneVsRest(:yes), [:yes,:no,:no,:maybe,:yes,:yes])
   6-element Array{Symbol,1}:
    :yes
    :not_yes
    :not_yes
    :not_yes
    :yes
    :yes

   julia> convertlabel(LabelEnc.ZeroOne, [:yes,:no,:no,:maybe,:yes,:yes], LabelEnc.OneVsRest(:yes))
   6-element Array{Float64,1}:
    1.0
    0.0
    0.0
    0.0
    1.0
    1.0

Some encodings come with an implicit contract of how the raw
predictions of some model should look like and how to classify a
raw prediction into a predicted class-label.
For that purpose we provide the function :func:`classify` and its
mutating version :func:`classify!`.

For :class:`LabelEnc.ZeroOne` the convention is that the raw
prediction is between 0 and 1 and represents a degree of
certainty that the observation is of the positive class. That
means that in order to classify a raw prediction to either
positive or negative, one needs to define a "threshold"
parameter, which determines at which degree of certainty a
prediction is "good enough" to classify as positive.

.. code-block:: jlcon

   julia> classify(0.3f0, 0.5); # equivalent to below
   julia> classify(0.3f0, LabelEnc.ZeroOne) # preserves type
   0.0f0

   julia> classify(0.3f0, LabelEnc.ZeroOne(0.5)) # defaults to Float64
   0.0

   julia> classify(0.3f0, LabelEnc.ZeroOne(Int,0.2))
   1

   julia> classify.([0.3,0.5], LabelEnc.ZeroOne(Int,0.4))
   2-element Array{Int64,1}:
    0
    1

For :class:`LabelEnc.MarginBased` on the other hand the decision
boundary is predefined at 0, meaning that any raw prediction greater
than or equal to zero is considered a positive prediction, while any
negative raw prediction is considered a negative prediction.

.. code-block:: jlcon

   julia> classify(0.3f0, LabelEnc.MarginBased) # preserves type
   1.0f0

   julia> classify(-0.3f0, LabelEnc.MarginBased()) # defaults to Float64
   -1.0

   julia> classify.([-2.3,6.5], LabelEnc.MarginBased(Int))
   2-element Array{Int64,1}:
    -1
     1

The encoding :class:`LabelEnc.OneOfK` is special in that it is
matrix-based and thus there exists the concept of ``ObsDim``,
i.e. the freedom to choose which array dimension denotes the
observations.
The classified prediction will be the index of the largest element of
an observation. By default the "obsdim" is defined as the last array
dimension.

.. code-block:: jlcon

   julia> pred_output = [0.1 0.4 0.3 0.2; 0.8 0.3 0.6 0.2; 0.1 0.3 0.1 0.6]
   3×4 Array{Float64,2}:
    0.1  0.4  0.3  0.2
    0.8  0.3  0.6  0.2
    0.1  0.3  0.1  0.6

   julia> classify(pred_output, LabelEnc.OneOfK)
   4-element Array{Int64,1}:
    2
    1
    2
    3

   julia> classify(pred_output', LabelEnc.OneOfK, obsdim=1) # note the transpose
   4-element Array{Int64,1}:
    2
    1
    2
    3

   julia> classify([0.1,0.2,0.6,0.1], LabelEnc.OneOfK) # single observation
   3

How to ... ?
-------------

Chances are you ended up here with a very specific use-case in mind.
This section outlines a number of different but common scenarios and
links to places that explain how this or a related package can be
utilized to solve them.

**What MLLabelUtils can do**

- :ref:`Infer which encoding some classification targets use.
  <infer>`

- :ref:`Assert if some classification targets are of the encoding I
  need them in. <assert>`

- :ref:`Convert targets into a specific encoding that my model
  requires. <convert>`

- :ref:`Group observations according to their class-label.
  <group>`

- :ref:`Work with matrices in which the user can choose of the
  rows or the columns denote the observations. <obsdim>`

- :ref:`Classify model predictions into class labels appropriate
  for the encoding of the targets. <classify>`

**What MLLabelUtils can NOT do** (outsourced)

- `Compute classification metrics, such as accuracy or a confusion
  matrix <https://github.com/JuliaML/MLMetrics.jl>`_

- `Compute margin-based loss functions, such as the hinge loss
  <http://lossesjl.readthedocs.io/en/latest/>`_

Getting Help
-------------

To get help on specific functionality you can either look up the
information here, or if you prefer you can make use of Julia's
native doc-system.
The following example shows how to get additional information on
:class:`LabelEnc.OneOfK` within Julia's REPL:

.. code-block:: julia

   ?LabelEnc.OneOfK

If you find yourself stuck or have other questions concerning the
package you can find us at gitter or the *Machine Learning*
domain on discourse.julialang.org

- `Julia ML on Gitter <https://gitter.im/JuliaML/chat>`_

- `Machine Learning on Julialang <https://discourse.julialang.org/c/domain/ML>`_

If you encounter a bug or would like to participate in the
further development of this package come find us on Github.

- `JuliaML/MLLabelUtils.jl <https://github.com/JuliaML/MLLabelUtils.jl>`_

