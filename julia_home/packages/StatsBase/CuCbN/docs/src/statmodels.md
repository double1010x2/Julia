# Abstraction for Statistical Models

This package defines an abstract type `StatisticalModel`, and an abstract subtype `RegressionModel`.

Particularly, instances of `StatisticalModel` implement the following methods.

```@docs
adjr2
aic
aicc
bic
coef
coefnames
coeftable
confint
deviance
dof
fit
fit!
informationmatrix
isfitted
islinear
loglikelihood
mss
nobs
nulldeviance
nullloglikelihood
r2
rss
score
stderror
vcov
weights(::StatisticalModel)
```

`RegressionModel` extends `StatisticalModel` by implementing the following additional methods.
```@docs
crossmodelmatrix
dof_residual
fitted
leverage
cooksdistance
meanresponse
modelmatrix
response
responsename
predict
predict!
residuals
```

An exception type is provided to signal convergence failures during model estimation:
```@docs
ConvergenceException
```