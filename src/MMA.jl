# This module implements the MMA Algorithm in Julia
# as described in:
# AU  - Svanberg, Krister
# TI  - The method of moving asymptotes—a new method for structural optimization
# JO  - International Journal for Numerical Methods in Engineering
# JA  - Int. J. Numer. Meth. Engng.
module MMA

using Parameters, StructsOfArrays

import Optim
import Optim: OnceDifferentiable, Fminbox, GradientDescent, update!, 
                MultivariateOptimizationResults, OptimizationTrace, maxdiff, 
                LineSearches, ConjugateGradient, LBFGS, AbstractOptimizer
import Base: min, max, show

export MMAModel, box!, ineq_constraint!, optimize

include("utils.jl")
include("model.jl")
include("primal.jl")
include("dual.jl")
include("trace.jl")

struct MMA87 <: AbstractOptimizer end

const μ = 0.1

function optimize(m::MMAModel{T,TV}, x0::TV, optimizer=ConjugateGradient(); s_init=T(0.5), s_incr=T(1.05), s_decr=T(0.65)) where {T, TV}
    check_error(m, x0)
    n_i = length(constraints(m))
    n_j = dim(m)
    x, x1, x2 = copy(x0), copy(x0), copy(x0)
    TM = MatrixOf(TV)

    # Buffers for bounds and move limits
    α, β, L, U = zerosof(TV, n_j), zerosof(TV, n_j), zerosof(TV, n_j), zerosof(TV, n_j)

    # Buffers for p0, pji, q0, qji
    p0, q0 = zerosof(TV, n_j), zerosof(TV, n_j)
    p, q = zerosof(TM, n_j, n_i), zerosof(TM, n_j, n_i)
    r = zerosof(TV, n_i)
    
    # Initial data and bounds for Optim to solve dual problem
    λ = onesof(TV, n_i)
    l = zerosof(TV, n_i)
    u = infsof(TV, n_i)

    # Objective gradient buffer
    ∇f_x = nansof(TV, length(x))
    g = zerosof(TV, n_i)
    ∇g = zerosof(TM, n_j, n_i)
    
    f_x::T = eval_objective(m, x, ∇f_x)
    f_calls, g_calls = 1, 1
    f_x_previous = T(NaN)

    # Evaluate the constraints and their gradients
    map!((i)->eval_constraint(m, i, x, (@view ∇g[:,i])), g, 1:n_i)

    # Build a primal data struct storing all primal problem's info
    primal_data = PrimalData(L, U, α, β, p0, q0, p, q, r, Ref(zero(T)), x, Ref(f_x), g, ∇f_x, ∇g)
    
    tr = OptimizationTrace{T, MMA87}()
    tracing = (m.store_trace || m.extended_trace || m.show_trace)

    converged = false
    # Assess multiple types of convergence
    x_converged, f_converged, gr_converged = false, false, false
    f_increased = false
    x_residual = T(Inf)
    f_residual = T(Inf)
    gr_residual = T(Inf)

    asymptotes_updater = AsymptotesUpdater(m, L, U, x, x1, x2, s_init, s_incr, s_decr)
    variable_bounds_updater = VariableBoundsUpdater(primal_data, m, μ)
    cvx_grad_updater = ConvexApproxGradUpdater(primal_data, m)

    x_updater = XUpdater(primal_data)
    dual_obj = DualObjVal(primal_data, λ, x_updater)
    dual_obj_grad = DualObjGrad(primal_data, x_updater)

    # Iteraton counter
    k = 0
    while !converged && k < m.max_iters
        k += 1
        asymptotes_updater(Iteration(k))

        # Track trial points two steps back
        copy!(x2, x1)
        copy!(x1, x)

        # Update convex approximation
        ## Update bounds on primal variables
        variable_bounds_updater()    

        ## Computes values and updates gradients of convex approximations of objective and constraints
        cvx_grad_updater()

        # Solve dual
        d = OnceDifferentiable(dual_obj, dual_obj_grad, λ)
        results = Optim.optimize(d, l, u, λ, Fminbox(optimizer), 
            Optim.Options(x_tol=xtol(m), f_tol=ftol(m), g_tol=grtol(m)))
        copy!(λ, results.minimizer::TV)

        x_updater(λ)

        # Evaluate the objective function and its gradient
        f_x_previous, f_x = f_x, eval_objective(m, x, ∇f_x)
        f_calls, g_calls = f_calls + 1, g_calls + 1
        # Correct for functions whose gradients go to infinity at some points, e.g. √x
        while mapreduce((x)->(isinf(x) || isnan(x)), or, false, ∇f_x)
            x .= T(0.01) .* x1 .+ T(0.99) .* x
            f_x = eval_objective(m, x, ∇f_x)
            f_calls, g_calls = f_calls + 1, g_calls + 1
        end
        primal_data.f_val[] = f_x

        # Evaluate the constraints and their Jacobian
        map!((i)->eval_constraint(m, i, x, (@view ∇g[:,i])), g, 1:n_i)

        # Assess convergence
        x_converged, f_converged, gr_converged, 
        x_residual, f_residual, gr_residual, 
        f_increased, converged = assess_convergence(x, x1, f_x, f_x_previous, ∇f_x, 
            xtol(m), ftol(m), grtol(m))

        # Print some trace if flag is on
        @mmatrace()
        if converged
            break
        end
    end
    h_calls = 0
    return MultivariateOptimizationResults{MMA87, T, TV, typeof(x_residual), typeof(f_x), typeof(tr)}(MMA87(),
        x0,
        x,
        f_x,
        k,
        k == m.max_iters,
        x_converged,
        xtol(m),
        x_residual,
        f_converged,
        ftol(m),
        f_residual,
        gr_converged,
        grtol(m),
        gr_residual,
        f_increased,
        tr,
        f_calls,
        g_calls,
        h_calls)
end

end # module
