module MMA

using Optim

import Base: min, max

export MMAModel, box!, ineq_constraint!, objective, dim, solve, constraint

include("checks.jl")

immutable MMAModel
    dim::Int
    objective::Function
    ineq_constraints::Vector{Function}
    box_max::Vector{Float64}
    box_min::Vector{Float64}

    # Store criterias
    store_obj::Bool
    store_x::Bool
    store_λ::Bool

    # Stopping criterias
    max_iters::Int
    ftol::Float64
    xtol::Float64
end

dim(m::MMAModel) = m.dim
min(m::MMAModel, i::Int) = m.box_min[i]
max(m::MMAModel, i::Int) = m.box_max[i]
min(m::MMAModel) = m.box_max
max(m::MMAModel) = m.box_min
objective(m::MMAModel) = m.objective
constraint(m::MMAModel) = m.ineq_constraints
constraint(m::MMAModel, i::Int) = m.ineq_constraints[i]
eval_objective(m, g0, ∇g0) = m.objective(g0, ∇g0)
eval_constraint(m, i, g0, ∇g0) = constraint(m, i)(g0, ∇g0)
ftol(m) = m.ftol
xtol(m) = m.xtol

function MMAModel(dim::Int,
                  objective::Function;
                  max_iters=200,
                  xtol::Float64 = eps(Float64),
                  ftol::Float64 = sqrt(eps(Float64)),
                  store_obj::Bool = false,
                  store_x::Bool = false,
                  store_λ::Bool = false)

    # TODO: Is this a good idea:
    mins = NaN * ones(dim)
    maxs = NaN * ones(dim)
    MMAModel(dim, objective, Function[],
             mins, maxs, store_obj, store_x, store_λ,
             max_iters, ftol, xtol)
end

type MMAResults
    n_iters::Int
    x0::Vector{Float64}
    obj_value::Float64
    minimum::Vector{Float64}
    f_history::Vector{Float64}
    x_history::Vector{Vector{Float64}}
    λ_history::Vector{Vector{Float64}}

    xtol::Float64
    ftol::Float64

    # Reason for stopping
    stopped_max_iter::Bool
    stopped_ftol::Bool
    stopped_xtol::Bool
end

function MMAResults(m, x)
    MMAResults(0, copy(x), NaN, Float64[], Float64[],
                          Array{Array{Float64,1}, 1}[], Array{Array{Float64,1}, 1}[],
                          xtol(m), ftol(m),
                          false, false, false)
end

function Base.show(io::IO, r::MMAResults)
    @printf io "Results of MMA Algorithm\n"
    @printf io " * Starting Point: [%s]\n" join(r.x0, ",")
    @printf io " * Minimum: [%s]\n" join(r.minimum, ",")
    @printf io " * Value of Function at Minimum: %f\n" r.obj_value
    @printf io " * Iterations: %d\n" r.n_iters
    @printf io " * Convergence: %s\n" r.stopped_ftol || r.stopped_xtol
    @printf io "   * |x - x'| < %.1e: %s\n" r.xtol r.stopped_xtol
    @printf io "   * |f(x) - f(x')| / |f(x)| < %.1e: %s\n" r.ftol r.stopped_ftol
    return
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

function solve(m::MMAModel, x::Vector{Float64})
    check_error(m, x)
    n_i = length(constraint(m))
    n_j = dim(m)
    x_k1 = copy(x)
    x_k2 = copy(x)
    α = zeros(dim(m))
    β = zeros(dim(m))
    L = zeros(dim(m))
    U = zeros(dim(m))
    n_ineqs = length(m.ineq_constraints)
    ∇fi = zeros(dim(m))
    ri = 0.0

    mma_results = MMAResults(m, x)


    # For box search
    λ = ones(n_ineqs)
    l = zeros(n_ineqs)
    u = Inf * ones(n_ineqs)

    mma_results.n_iters = 0
    while true
        mma_results.n_iters += 1
        k = mma_results.n_iters
        ri = eval_objective(m, x, ∇fi)
        mma_results.obj_value = ri
        mma_results.minimum = x
        # Check convergence.
        # TODO: Refactor into it's own function
        if k > 1
            if k >= m.max_iters
                mma_results.stopped_max_iter = true
                return mma_results
            elseif Optim.maxdiff(x, x_k1) < xtol(m)
                mma_results.stopped_xtol = true
                return mma_results
            elseif abs(ri - f_k1) / (abs(ri) + ftol(m)) < ftol(m)
                mma_results.stopped_ftol = true
                return mma_results
            end
        end

        copy!(x_k2, x_k1)
        copy!(x_k1, x)
        f_k1 = ri

        r0 = 0.0
        r = zeros(n_i)

        p0 = zeros(n_j)
        q0 = zeros(n_j)
        p = zeros(n_i, n_j)
        q = zeros(n_i, n_j)

        update_L_U!(L, U, m, k, x_k2, x_k1, x)
        for j = 1:dim(m)
            α[j] = max(0.9 * L[j] + 0.1 * x[j], min(m, j))
            β[j] = min(0.9 * U[j] + 0.1 * x[j], max(m, j))
        end

        for (i, f) in enumerate([objective(m); constraint(m)])
            # Also computes the tangent and stores it in second argument
            if i > 1
                ri = f(x, ∇fi)
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
                if i == 1
                    p0[j] = p_ij
                    q0[j] = q_ij
                else
                    p[i-1, j] = p_ij
                    q[i-1, j] = q_ij
                end
            end
            if i == 1
                r0 = ri
            else
                r[i-1] = ri
            end
        end

        φ = (λ) -> compute_φ(λ, Float64[], x, r, r0, p, p0, q, q0, L, U, α, β)
        φgrad = (λ, grad) -> compute_φ(λ, grad, x, r, r0, p, p0, q, q0, L, U, α, β)
        d4 = DifferentiableFunction(φ, φgrad, φgrad)
        results = fminbox(d4, λ, l, u)
        λ = results.minimum

        compute_x!(x, λ, p, p0, q, q0, L, U, α, β)
    end

    return x
end

function compute_φ(λ, ∇f, x, r, r0, p, p0, q, q0, L, U, α, β)
    φ = r0 + dot(λ, r)
    compute_x!(x, λ, p, p0, q, q0, L, U, α, β)
    for j = 1:length(x)
        Ujxj = U[j] - x[j]
        xjLj = x[j] - L[j]
        φ += (p0[j] + dot(λ, p[:,j])) / Ujxj
        φ += (q0[j] + dot(λ, q[:,j])) / xjLj
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

function update_L_U!(L, U, m, k, x_k2, x_k1, x_k)
    s = 0.7
    for j = 1:dim(m)
        if k == 1 || k == 2
            L[j] = x_k[j] - (max(m,j) - min(m, j))
            U[j] = x_k[j] + (max(m,j) - min(m, j))
        else
            if sign(x_k[j] - x_k1[j]) != sign(x_k1[j] - x_k2[j])
                L[j] = x_k[j] - (x_k1[j] - L[j]) * s
                U[j] = x_k[j] + (U[j] - x_k1[j]) * s
            else
                L[j] = x_k[j] - (x_k1[j] - L[j]) / s
                U[j] = x_k[j] + (U[j] - x_k1[j]) / s
            end
        end
    end
end

end # module
