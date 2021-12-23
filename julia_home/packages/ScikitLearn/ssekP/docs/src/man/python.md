# Reading the Scikit-Learn Documentation.

## Relationship to scikit-learn

ScikitLearn.jl aims to mirror the Python scikit-learn project, but the API had to be adapted to Julia, and follows Julia's conventions. When reading the Python [documentation](http://scikit-learn.org/stable/documentation.html), keep in mind:

- Most object methods are now functions: Python's `model.predict(X)` becomes
  `predict(model, X)`
- Methods that modify the model's state have a `!` at the end:
  `model.fit_transform(X)` becomes `fit_transform!(model, X)`
- A few of the Python submodules were translated into Julia to support
  Julia models: `ScikitLearn.Pipelines`, `ScikitLearn.CrossValidation`, and `ScikitLearn.GridSearch`


You can access the class members and methods of a Python object
(i.e. all models imported through `@sk_import`) using `obj.member_name`. For
example:

```@repl
using ScikitLearn #hide
X=rand(10,4);y=rand(10);

@sk_import linear_model: Lasso
lm = fit!(Lasso(), X, y)
println(lm.n_iter_)

```

This is rarely necessary, because the most important/frequently-used methods
have been defined in Julia (eg. `transformer.classes_` is now
`get_classes(transformer)`)
