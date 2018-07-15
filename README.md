# MMA.jl

[![Build Status](https://travis-ci.org/KristofferC/MMA.jl.svg?branch=master)](https://travis-ci.org/KristofferC/MMA.jl)

This module implements the MMA Algorithm in Julia as described by Krister Svanberg in [1] and [2].

## Usage

Ẁe will solve the nonlinear constrained minimization problem given [here](http://ab-initio.mit.edu/wiki/index.php/NLopt_Tutorial)

```julia
using MMA

# Define objective function
function f(x::AbstractVector, grad::AbstractVector)
    if length(grad) != 0
        grad[1] = 0.0
        grad[2] = 0.5/sqrt(x[2])
    end
    sqrt(x[2])
end

# Define a constraint function
function g(x::AbstractVector, grad::AbstractVector, a, b)
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
#results = optimize(m, x0, MMA.MMA02())  #-> Default
results = optimize(m, x0, MMA.MMA87())

# Print the results
print(results)
#Results of Optimization Algorithm
# * Algorithm: MMA.MMA87
# * Starting Point: [1.234,2.345]
# * Minimizer: [0.3333341071616548,0.29629732757492855]
# * Minimum: 5.443320e-01
# * Iterations: 7
# * Convergence: true
#   * |x - x'| ≤ 1.0e-06: true
#     |x - x'| = 7.75e-07
#   * |f(x) - f(x')| ≤ 1.5e-08 |f(x)|: false
#     |f(x) - f(x')| = 1.31e-06 |f(x)|
#   * |g(x)| ≤ 1.5e-08: false
#     |g(x)| = 9.19e-01
#   * Stopped by an increasing objective: true
#   * Reached Maximum Number of Iterations: false
# * Objective Calls: 8
# * Gradient Calls: 8

# Print the trace
println(results.trace)
#Iter     Function value   Gradient norm
#------   --------------   --------------
#     1     3.947968e-01     1.266474e+00
#     2     1.803133e-01     2.772951e+00
#     3     4.353112e-01     1.148604e+00
#     4     5.338189e-01     9.366472e-01
#     5     5.442496e-01     9.186962e-01
#     6     5.443313e-01     9.185583e-01
#     7     5.443320e-01     9.185571e-01
```

## References
[1] [The method of moving asymptotes - a new method for structural optimization](http://www.researchgate.net/publication/227631828_The_method_of_moving_asymptotesa_new_method_for_structural_optimization)

[2] [A class of globally convergent optimization methods based on conservative convex separable approximations](https://epubs.siam.org/doi/10.1137/S1052623499362822)

### Authors
Kristoffer Carlsson - kristoffer.carlsson@chalmers.se

Mohamed Tarek - mohamed82008@gmail.com
