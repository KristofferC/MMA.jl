module MMA

using Optim

import Base: min, max

export MMAModel, box!, ineq_constraint!, objective, dim, solve, constraint



immutable MMAModel
    dim::Int
    objective::Function # Should return g(x), δg(x)/δx
    ineq_constraints::Vector{Function}
    box_max::Vector{Float64}
    box_min::Vector{Float64}
    max_iters::Int
    store_f_history::Bool
    store_x_history::Bool
    #stopval::Float64
    #ftol_rel::Float64
    #ftol_abs::Float64
    #xtol_abs::Float64
    #xtol_rel::Float64
end

immutable MMAResults
    n_iters::Int
    obj_value::Float64
    f_history::Vector{Float64}
    x_history::Vector{Float64}
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
eval_constraint(m, i, g0, ∇g0) = m.box_constraints[i](g0, ∇g0)

function MMAModel(dim::Int, objective::Function; max_iters=30)
    mins = zeros(dim)
    maxs = ones(dim) * Inf
    MMAModel(dim, objective, Vector{Function}(), mins, maxs, max_iters,false, false)
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
    error_check(m, x)
    n_i = length(constraint(m))
    n_j = dim(m)

    L = zeros(dim(m))
    U = 10 * m.box_max
    α = 0.9 * L + 0.1 * x
    β = 0.9 * U + 0.1 * x
    #update_L_U!(L, U, m, k, x, prev_x)

    n_ineqs = length(m.ineq_constraints)

    for k = 1:3
        ∇fi = zeros(dim(m))

        r0 = 0.0
        r = zeros(n_i)

        p0 = zeros(n_j)
        q0 = zeros(n_j)
        p = zeros(n_i, n_j)
        q = zeros(n_i, n_j)

       # for k = 1:max_iters
       #     x_k2 = x_k1
        #    x_k1 = x

        for (i, f) in enumerate([objective(m); constraint(m)])
            fi = f(x, ∇fi)
            #println("∇fi_mma: $∇fi")
            rv = fi
            for j in 1:dim(m)
                Ujxj = U[j] - x[j]
                xjLj = x[j] - L[j]
                if ∇fi[j] > 0
                    pv = Ujxj^2 * ∇fi[j]
                    qv = 0.0
                else
                    pv = 0.0
                    qv = -xjLj^2 * ∇fi[j]
                end
                rv -= pv / Ujxj + qv / xjLj
                if i == 1
                    p0[j] = pv
                    q0[j] = qv
                else
                    p[i-1, j] = pv
                    q[i-1, j] = qv
                end
            end
            if i == 1
                r0 = rv
            else
                r[i-1] = rv
            end
        end


    #compute_x!(x, λ, p, q, L, U, α, β)
    φ = (λ) -> compute_φ(λ, Float64[], x, r, r0, p, p0, q, q0, L, U, α, β)
    φgrad = (λ, grad) -> compute_φ(λ, grad, x, r, r0, p, p0, q, q0, L, U, α, β)

    d4 = DifferentiableFunction(φ, φgrad)

    l = zeros(n_ineqs)
    u = [Inf]
    x0 = [1.0]
    results = fminbox(d4, x0, l, u)

end

    return x
end

function compute_φ(λ, grad, x, r, r0, p, p0, q, q0, L, U, α, β)
    λ = [λ]
    b = [0.008188]
    φ = r0 - dot(λ, b)
 #   println("r: $r")
 #   println("r0: $r0")
    #println(r)
   # println("r: $r")
    compute_x!(x, λ, p, p0, q, q0, L, U, α, β)
   # println("x: $x")
   # println("q0: $q0")
   # println("$U")
    for j = 1:length(x)
        Ujxj = U[j] - x[j]
        xjLj = x[j] - L[j]
        φ += q0[j] / (x[j] - L[j]) + dot(λ, p[:,j])*x[j]
        #φ += (p0[j] + dot(λ, p[:,j])) / (U[j] - x[j])
        #φ += (q0[j] + dot(λ, q[:,j])) / (x[j] - L[j])
        #println("$Ujxj")
        #φ += (λ'*p[:,j]) / Ujxj
    end

    if length(grad) != 0
        for i = 1:length(λ)
            grad[i] = -b[i]
            for j = 1:length(x)
                grad[i]  += p[i,j] * x[j]
                #grad[i] += p[i,j] / (U[j] - x[j])
                #grad[i] += q[i,j] / (x[j] - L[j])
            end
        end
    end
    #println("φ: $φ")
    scale!(grad, -1.0)
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
    #println("x_mma: $x")
end

function update_L_U!(L, U, m, s, k, x_k2, x_k1, x_k)
    s = 0.7
    for j = 1:dim(m)
        if k == 1 || k == 2
            L[j] = x_k[j] - (max(m,j) - min(m, i))
            U[j] = x_k[j] + (max(m,j) - min(m, i))
        else
            if sign(x_k[j] - x_k1[j]) != sign(x_k1[j] - x_k2[j])
                L[j] = x_k[j] - s * (x_k1[j] - L[j])
                U[j] = x_k[j] + s * (U[j] - x_k1[j])
            else
                L[j] = x_k[j] - (x_k1[j] - L[j]) / s
                U[j] = x_k[j] + (U[j] - x_k1[j]) / s
            end
        end
    end
end



function error_check(m, x0)
    if length(x0) != dim(m)
        throw(ArgumentError("initial variable must have same dimension as model"))
    end

    for (i, x) in enumerate(x0)
        # x is not in box
        if !(min(m, i) <= x <= max(m,i))
            throw(ArgumentError("initial variable at index $i outside box constraint"))
        end
    end
    for g in m.ineq_constraints
        # x0 is outside ineq constraint
        println(g(x0, []))
        if g(x0, []) > 0
            throw(ArgumentError("initial variable outside inequality constraint"))
        end
    end
end

end # module
