# MMA.jl

[![Build Status](https://travis-ci.org/KristofferC/MMA.jl.svg?branch=master)](https://travis-ci.org/KristofferC/MMA.jl)

This module implements the MMA Algorithm in Julia as described by Krister Svanberg in [1].

The code in this module was made for a course in Structural Optimization and should be seen as educational. For real use it is likely better to use a more mature code base, for example [NLopt.jl](https://github.com/JuliaOpt/NLopt.jl) which contain a more modern MMA algorithm than the one implemented here.

## Usage

Ẁe will solve the nonlinear constrained minimization problem given [here](http://ab-initio.mit.edu/wiki/index.php/NLopt_Tutorial)

```julia
# Define objective function
function f(x::Vector, grad::Vector)
    if length(grad) != 0
        grad[1] = 0.0
        grad[2] = 0.5/sqrt(x[2])
    end
    sqrt(x[2])
end

# Define a constraint function
function g(x::Vector, grad::Vector, a, b)
    if length(grad) != 0
        grad[1] = 3a * (a*x[1] + b)^2
        grad[2] = -1
    end
    (a*x[1] + b)^3 - x[2]
end

# Create the MMAModel with a relative tolerance on x
m = MMAModel(2, f, xtol = 1e-4)

# Add box constraints to the variables
box!(m, 1, 0.0, 100.0)
box!(m, 2, 0.0, 100.0)

# Add two nonlinear inequalities
ineq_constraint!(m, (x,grad) -> g(x,grad,2,0))
ineq_constraint!(m, (x,grad) -> g(x,grad,-1,1))

# Solve the problem
results = solve(m, [0.5, 5.0])

# Print the results
print(results)
#Results of Optimization Algorithm
# * Algorithm: MMA
# * Starting Point: [0.5,5.0]
# * Minimum: [0.33333334126240005,0.29629630317993827]
# * Value of Function at Minimum: 0.544331
# * Iterations: 12
# * Convergence: true
#   * |x - x'| < 1.0e-08: true
#   * |f(x) - f(x')| / |f(x)| < 1.0e-10: false
#   * |g(x)| < 1.5e-08: false
#   * Exceeded Maximum Number of Iterations: false
# * Objective Function Calls: 13
# * Gradient Call: 13
```


## References
[1] [The method of moving asymptotes - a new method for structural optimization](http://www.researchgate.net/publication/227631828_The_method_of_moving_asymptotesa_new_method_for_structural_optimization)

### Author
Kristoffer Carlsson - kristoffer.carlsson@chalmers.se
