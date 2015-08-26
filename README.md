# MMA.jl

[![Build Status](https://travis-ci.org/KristofferC/MMA.jl.svg?branch=master)](https://travis-ci.org/KristofferC/MMA.jl)

This module implements the MMA Algorithm in Julia as described by Krister Svanberg in [1].

The code in this module was made for a course in Structural Optimization and should be seen as educational. For real use it is likely better to use a more mature code base, for example [NLopt.jl](https://github.com/JuliaOpt/NLopt.jl) which contain a more modern MMA algorithm than the one implemented here.

## Usage

Ẁe will solve the nonlinear constrained minimization problem given [here](http://ab-initio.mit.edu/wiki/index.php/NLopt_Tutorial)

```julia
using MMA

# Define objective function
function f(x::Vector, grad::AbstractArray)
    if length(grad) != 0
        grad[1] = 0.0
        grad[2] = 0.5/sqrt(x[2])
    end
    sqrt(x[2])
end

# Define a constraint function
function g(x::Vector, grad::AbstractArray, a, b)
    if length(grad) != 0
        grad[1] = 3a * (a*x[1] + b)^2
        grad[2] = -1
    end
    (a*x[1] + b)^3 - x[2]
end

# Create the MMAModel with a relative tolerance on x
ndim = 2
m = MMAModel(ndim, f, xtol = 1e-6, store_trace=true)

# Add box constraints to the variables
box!(m, 1, 0.0, 100.0)
box!(m, 2, 0.0, 100.0)

# Add two nonlinear inequalities
ineq_constraint!(m, (x,grad) -> g(x,grad,2,0))
ineq_constraint!(m, (x,grad) -> g(x,grad,-1,1))

# Solve the problem
x0 = [1.234, 2.345]
results = optimize(m, x0)

# Print the results
print(results)

# Results of Optimization Algorithm
#  * Algorithm: MMA
#  * Starting Point: [1.234,2.345]
#  * Minimum: [0.3333335414323581,0.2962963019993659]
#  * Value of Function at Minimum: 0.544331
#  * Iterations: 7
#  * Convergence: true
#    * |x - x'| < 1.0e-06: true
#    * |f(x) - f(x')| / |f(x)| < 1.5e-08: false
#    * |g(x)| < 1.5e-08: false
#    * Exceeded Maximum Number of Iterations: false
#  * Objective Function Calls: 8
#  * Gradient Call: 8

# Print the trace
#print(results.trace)
#Iter     Function value   Gradient norm
#------   --------------   --------------
#     1     8.833999e-01     5.659951e-01
#     2     3.766758e-01     1.327401e+00
#     3     4.672562e-01     1.070077e+00
#     4     5.408004e-01     9.245555e-01
#     5     5.443183e-01     9.185802e-01
#     6     5.443318e-01     9.185573e-01
#     7     5.443311e-01     9.185586e-01
```

## References
[1] [The method of moving asymptotes - a new method for structural optimization](http://www.researchgate.net/publication/227631828_The_method_of_moving_asymptotesa_new_method_for_structural_optimization)

### Author
Kristoffer Carlsson - kristoffer.carlsson@chalmers.se
