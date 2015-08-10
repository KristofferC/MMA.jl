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
import Base: min, max, show

export MMAModel, box!, ineq_constraint!, optimize

include("utils.jl")

macro mmatrace()
    quote
        if tracing
            dt = Dict()
            if m.extended_trace
                dt["x"] = copy(x)
                dt["g(x)"] = copy(∇f_x)
                dt["λ"] = copy(results.minimum)
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

type DualData
    L::Vector{Float64}
    U::Vector{Float64}
    α::Vector{Float64} # Lower move limit
    β::Vector{Float64} # Upper move limit
    p0::Vector{Float64}
    q0::Vector{Float64}
    p::Matrix{Float64}
    q::Matrix{Float64}
    r::Vector{Float64}
    r0::Float64
    x::Vector{Float64} # Optimal value for x in dual iteration
    f_val::Float64 # Function value at current iteration
    g_val::Vector{Float64} # Inequality values at current iteration
    ∇f::Vector{Float64} # Function gradient at current iteration
    ∇g::Matrix{Float64} # Inequality gradients [ineq][var] at current iteration
end

# Inspired by Parameters.jl
const dd_fields = [:L, :U, :α, :β, :p0, :q0, :p, :q,
                   :r, :r0, :x, :f_val, :g_val, :∇f, :∇g]
macro unpack(dual_data)
    esc(Expr(:block, [:($f = dual_data.$f) for f in dd_fields]...))
end

function show(io::IO, dual_data::DualData)
    println("Dual data:")
    for f in dd_fields
        println("$f = $(dual_data.(f))")
    end
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
    x0, x, x1, x2 = copy(x0), copy(x0), copy(x0), copy(x0)

    # Buffers for bounds and move limits
    α, β, L, U = zeros(n_j), zeros(n_j), zeros(n_j), zeros(n_j)

    # Buffers for p0, pij, q0, qij
    p0, q0 = zeros(n_j), zeros(n_j)
    p, q = zeros(n_i, n_j), zeros(n_i, n_j)
    r = zeros(n_i)

    # Initial data and bounds for fminbox to solve dual problem
    λ = ones(n_i)
    l = zeros(n_i)
    u = Inf * ones(n_i)

    # Objective gradient buffer
    ∇f_x = similar(x)
    f_x = eval_objective(m, x, ∇f_x)
    f_calls, g_calls = 1, 1
    f_x_prev = NaN

    # Evaluate the constraints and their gradients
    g = zeros(n_i)
    ∇g = zeros(n_j, n_i)
    for i = 1:n_i
        g[i] = eval_constraint(m, i, x, sub(∇g, :, i))
    end

    # Create a DualData type that holds the data needed for the dual problem
    dual_data = DualData(L, U, α, β, p0, q0, p, q, r, 0.0, x, f_x, g, ∇f_x, ∇g)

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
        copy!(x2, x1)
        copy!(x1, x)

        update_limits!(dual_data, m, k, x1, x2)
        compute_mma!(dual_data, m)

        dual = (λ) -> compute_dual!(λ, Float64[], dual_data)
        dual_grad = (λ, grad_dual) -> compute_dual!(λ, grad_dual, dual_data)
        d = DifferentiableFunction(dual, dual_grad, dual_grad)
        results = fminbox(d, λ, l, u, xtol=1e-10, ftol=1e-10)
        f_x_previous, f_x = f_x, eval_objective(m, x, ∇f_x)
        f_calls, g_calls = f_calls + 1, g_calls + 1
         # Evaluate the constraints and their gradients
        for i = 1:n_i
            g[i] = eval_constraint(m, i, x, sub(∇g, :, i))
        end
        @mmatrace()

        x_converged, f_converged,
        gr_converged, converged = assess_convergence(x, x1, f_x, f_x_previous, ∇f_x,
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
function compute_mma!(dual_data, m)
    @unpack dual_data
    # Bound limits
    for j = 1:dim(m)
            α[j] = max(0.9 * L[j] + 0.1 * x[j], min(m, j))
            β[j] = min(0.9 * U[j] + 0.1 * x[j], max(m, j))
    end

    r0 = 0.0
    for i in 0:length(constraints(m))
        if i == 0
            ri = f_val
            ∇fi = ∇f
        else
             ri = g_val[i]
             ∇fi = sub(∇g, :, i)
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
    dual_data.r0 = r0
end

# Update move limits
function update_limits!(dual_data, m, k, x1, x2)
    @unpack dual_data
    for j in 1:dim(m)
        if k == 1 || k == 2
            # Equation 11 in Svanberg
            L[j] = x[j] - (max(m,j) - min(m, j))
            U[j] = x[j] + (max(m,j) - min(m, j))
        else
            # Equation 12 in Svanberg
            s = 0.7 # Suggested by Svanberg
            if sign(x[j] - x1[j]) != sign(x1[j] - x2[j])
                L[j] = x[j] - (x1[j] - L[j]) * s
                U[j] = x[j] + (U[j] - x1[j]) * s
            # Equation 13 in Svanberg
            else
                L[j] = x[j] - (x1[j] - L[j]) / s
                U[j] = x[j] + (U[j] - x1[j]) / s
            end
        end
    end
end


function compute_dual!(λ, ∇φ, dual_data)
    @unpack dual_data
    φ = r0 + dot(λ, r)
    update_x!(dual_data, λ)
    for j = 1:length(x)
        φ += (p0[j] + dot(λ, p[:,j])) / (U[j] - x[j])
        φ += (q0[j] + dot(λ, q[:,j])) / (x[j] - L[j])
    end

    if length(∇f) > 0
        for i = 1:length(λ)
            ∇φ[i] = r[i]
            for j = 1:length(x)
                ∇φ[i] += p[i,j] / (U[j] - x[j])
                ∇φ[i] += q[i,j] / (x[j] - L[j])
            end
        end
    end
    # Negate since we have a maximization problem
    scale!(∇φ, -1.0)
    return -φ
end

# Updates x to be the analytical optimal point in the dual
# problem for a given λ
function update_x!(dual_data, λ)
    @unpack dual_data
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
