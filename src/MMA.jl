# This module implements the MMA Algorithm in Julia
# as described in:
# AU  - Svanberg, Krister
# TI  - The method of moving asymptotes—a new method for structural optimization
# JO  - International Journal for Numerical Methods in Engineering
# JA  - Int. J. Numer. Meth. Engng.
module MMA

using Optim
using Compat

import Optim: update!, MultivariateOptimizationResults,
              OptimizationTrace
import Base: min, max

export MMAModel, box!, ineq_constraint!, optimize

include("utils.jl")

macro mmatrace()
    quote
        if tracing
            dt = Dict()
            if m.extended_trace
                dt["x"] = copy(x)
                dt["g(x)"] = copy(∇f_x)
                dt["λ"] = copy(λ)
            end
            grnorm = norm(∇f_x[:], Inf)
            update!(tr,
                    k,
                    f_x,
                    grnorm,
                    dt,
                    m.store_trace,
                    m.show_trace)
        end
    end
end

immutable MMAModel
    dim::Int
    objective::Function
    ineq_constraints::Vector{Function}
    box_max::Vector{Float64}
    box_min::Vector{Float64}

    store_trace::Bool
    show_trace::Bool
    extended_trace::Bool

    # Stopping criterias
    max_iters::Int
    ftol::Float64
    xtol::Float64
    grtol::Float64
end

dim(m::MMAModel) = m.dim
min(m::MMAModel, i::Int) = m.box_min[i]
max(m::MMAModel, i::Int) = m.box_max[i]
min(m::MMAModel) = m.box_max
max(m::MMAModel) = m.box_min
objective(m::MMAModel) = m.objective
constraints(m::MMAModel) = m.ineq_constraints
constraint(m::MMAModel, i::Int) = m.ineq_constraints[i]
eval_objective(m, g0, ∇g0) = m.objective(g0, ∇g0)
eval_objective(m, g0) = m.objective(g0, Float64[])
eval_constraint(m, i, g0, ∇g0) = constraint(m, i)(g0, ∇g0)
eval_constraint(m, i, g0) = constraint(m, i)(g0, Float64[])
ftol(m) = m.ftol
xtol(m) = m.xtol
grtol(m) = m.grtol
ftol!(m, v) = m.ftol = v
xtol!(m, v) = m.xtol = v
grtol!(m, v) = m.grtol = v

function MMAModel(dim::Int,
                  objective::Function;
                  max_iters=200,
                  xtol::Float64 = eps(Float64),
                  ftol::Float64 = sqrt(eps(Float64)),
                  grtol::Float64 = sqrt(eps(Float64)),
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false)

    mins = Inf * ones(dim)
    maxs = -Inf * ones(dim)
    MMAModel(dim, objective, Function[],
             mins, maxs, store_trace, show_trace, extended_trace,
             max_iters, ftol, xtol, grtol)
end

# Box constraints
function box!(m::MMAModel, i::Int, minb::Float64, maxb::Float64)
    if !(1 <= i <= dim(m))
        throw(ArgumentError("box constraint need to applied to an existing variable"))
    end
    m.box_min[i] = minb
    m.box_max[i] = maxb
end

function box!(m::MMAModel, minb::Float64, maxb::Float64)
    for i = 1:dim(m)
        m.box_min[i] = minb
        m.box_max[i] = maxb
    end
end

function box!(m::MMAModel, minbs::Vector{Float64}, maxbs::Vector{Float64})
    if (length(minbs) != dim(m)) || (length(minbs) != dim(m))
        throw(ArgumentError("box constraint vector must have same size as problem dimension"))
    end
    for i = 1:dim(m)
        m.box_min[i] = minbs[i]
        m.box_max[i] = maxbs[i]
    end
end

function ineq_constraint!(m::MMAModel, f::Function)
    push!(m.ineq_constraints, f)
end

function ineq_constraint!(m::MMAModel, fs::Vector{Function})
    for f in fs
        push!(m.ineq_constraints, f)
    end
end

function optimize(m::MMAModel, x0::Vector{Float64})
    check_error(m, x0)
    n_i = length(constraints(m))
    n_j = dim(m)
    x0, x, x_k1, x_k2 = copy(x0), copy(x0), copy(x0), copy(x0)


    # Buffers for bounds and move limits
    α, β, L, U = zeros(n_j), zeros(n_j), zeros(n_j), zeros(n_j)

    # Buffers for p0, pij, q0, qij
    p0, q0 = zeros(n_j), zeros(n_j)
    p, q = zeros(n_i, n_j), zeros(n_i, n_j)
    r = zeros(n_i)

    # Objective gradient buffer
    ∇f_x = similar(x)

    # Constraint gradient buffer
    ∇g = similar(x)

    # Buffers, initial data and bounds for fminbox to solve dual problem
    ∇φ = zeros(n_i)
    λ = ones(n_i)
    l = zeros(n_i)
    u = Inf * ones(n_i)

    # Start of with evaluating the function at x0
    f_x = eval_objective(m, x, ∇f_x)
    f_calls, g_calls = 1, 1
    f_x_prev = NaN

    tr = OptimizationTrace()
    tracing = m.store_trace || m.extended_trace || m.show_trace

    converged = false
    # Assess multiple types of convergence
    x_converged, f_converged, gr_converged = false, false, false

    # Iteraton counter
    k = 0
    while !converged && k < m.max_iters
        k += 1
        # Track trial points two steps back
        copy!(x_k2, x_k1)
        copy!(x_k1, x)

        update_limits!(L, U, m, k, x, x_k1, x_k2)
        L = zeros(size(U))
        r0 = compute_mma!(p0, q0, p, q, r, m, f_x, ∇f_x, ∇g, L, U, α, β, x)

        dual = (λ) -> compute_dual!(λ, Float64[], x, r, r0, p, p0, q, q0, L, U, α, β)
        dual_grad = (λ, grad_dual) -> compute_dual!(λ, grad_dual, x, r, r0, p, p0, q, q0, L, U, α, β)
        d = DifferentiableFunction(dual, dual_grad, dual_grad)
        results = fminbox(d, λ, l, u, xtol=1e-10, ftol=1e-10)
        λ = results.minimum
        f_x_previous, f_x = f_x, eval_objective(m, x, ∇f_x)
        f_calls, g_calls = f_calls + 1, g_calls + 1
        @mmatrace()

        x_converged, f_converged,
        gr_converged, converged = assess_convergence(x, x_k1, f_x, f_x_previous, ∇f_x,
                                                     xtol(m), ftol(m), grtol(m))
        if converged
            break
        end

    end

    return MultivariateOptimizationResults("MMA",
                                           x0,
                                           x,
                                           @compat(Float64(f_x)),
                                           k,
                                           k == m.max_iters,
                                           x_converged,
                                           xtol(m),
                                           f_converged,
                                           ftol(m),
                                           gr_converged,
                                           grtol(m),
                                           tr,
                                           f_calls,
                                           g_calls)
end

# Updates p0, q0, p, q, r and returns r0.
# For notation see reference at top
function compute_mma!(p0, q0, p, q, r, m, f_x, ∇f_k, ∇g, L, U, α, β, x)
    # Bound limits
    for j = 1:dim(m)
            α[j] = max(0.9 * L[j] + 0.1 * x[j], min(m, j))
            β[j] = min(0.9 * U[j] + 0.1 * x[j], max(m, j))
    end

    r0 = 0.0
    for i in 0:length(constraints(m))
        if i == 0
            ri = f_x
            ∇fi = ∇f_k
        else
             ri = eval_constraint(m, i, x, ∇g)
             ∇fi = ∇g
        end
        for j in 1:dim(m)
            Ujxj = U[j] - x[j]
            xjLj = x[j] - L[j]
            if ∇fi[j] > 0
                p_ij = abs2(Ujxj) * ∇fi[j]
                q_ij = 0.0
            else
                p_ij = 0.0
                q_ij = -abs2(xjLj) * ∇fi[j]
            end
            ri -= p_ij / Ujxj + q_ij / xjLj
            if i == 0
                p0[j] = p_ij
                q0[j] = q_ij
            else
                p[i, j] = p_ij
                q[i, j] = q_ij
            end
        end
        if i == 0
            r0 = ri
        else
            r[i] = ri
        end
    end
    return r0
end

# Update move limits
function update_limits!(L, U, m, k, x_k, x_k1, x_k2)
    for j in 1:dim(m)
        if k == 1 || k == 2
            # Equation 11 in Svanberg
            L[j] = x_k[j] - (max(m,j) - min(m, j))
            U[j] = x_k[j] + (max(m,j) - min(m, j))
        else
            # Equation 12 in Svanberg
            s = 0.7 # Suggested by Svanberg
            if sign(x_k[j] - x_k1[j]) != sign(x_k1[j] - x_k2[j])
                L[j] = x_k[j] - (x_k1[j] - L[j]) * s
                U[j] = x_k[j] + (U[j] - x_k1[j]) * s
            # Equation 13 in Svanberg
            else
                L[j] = x_k[j] - (x_k1[j] - L[j]) / s
                U[j] = x_k[j] + (U[j] - x_k1[j]) / s
            end
        end
    end
end

function compute_dual!(λ, ∇f, x, r, r0, p, p0, q, q0, L, U, α, β)
    φ = r0 + dot(λ, r)
    compute_x!(x, λ, p, p0, q, q0, L, U, α, β)

    for j = 1:length(x)
        φ += (p0[j] + dot(λ, p[:,j])) / (U[j] - x[j])
        φ += (q0[j] + dot(λ, q[:,j])) / (x[j] - L[j])
    end

    if length(∇f) > 0
        for i = 1:length(λ)
            ∇f[i] = r[i]
            for j = 1:length(x)
                ∇f[i] += p[i,j] / (U[j] - x[j])
                ∇f[i] += q[i,j] / (x[j] - L[j])
            end
        end
    end
    # Negate since we have a maximization problem
    scale!(∇f, -1.0)
    return -φ
end


function compute_x!(x, λ, p, p0, q, q0, L, U, α, β)
    for j in 1:length(x)
        fpj = sqrt((p0[j] + dot(λ, p[:,j])))
        fqj = sqrt((q0[j] + dot(λ, q[:,j])))
        x[j] = (fpj * L[j] + fqj * U[j]) / (fpj + fqj)
        if x[j] > β[j]
            x[j] = β[j]
        elseif x[j] < α[j]
            x[j] = α[j]
        end
    end
end

end # module
