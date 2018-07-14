struct PrimalData{T, TV<:AbstractVector{T}, TM<:AbstractMatrix{T}}
    L::TV
    U::TV
    α::TV # Lower move limit
    β::TV # Upper move limit
    p0::TV
    q0::TV
    p::TM
    q::TM
    r::TV
    r0::Base.RefValue{T}
    x::TV # Optimal value for x in dual iteration
    f_val::Base.RefValue{T} # Function value at current iteration
    g_val::TV # Inequality values at current iteration
    ∇f::TV # Function gradient at current iteration
    ∇g::TM # Inequality gradients [var, ineq] at current iteration
end
function show(io::IO, primal_data::PrimalData)
    println("Primal data:")
    for f in pd_fields
        println("$f = $(primal_data.(f))")
    end
end

struct XUpdater{TPD<:PrimalData}
    pd::TPD
end
function (xu::XUpdater)(λ)
    @unpack x = xu.pd
    map!((j)->xu(λ, j), x, 1:length(x))
end
function (xu::XUpdater)(λ, j)
    @unpack p0, p, q0, q, L, U, α, β = xu.pd
    lj1 = p0[j] + matdot(λ, p, j)
    lj2 = q0[j] + matdot(λ, q, j)
    αj = α[j]
    βj = β[j]
    Lj = L[j]
    Uj = U[j]

    Ujαj = U[j] - αj
    αjLj = αj - L[j]
    ljαj = lj1/Ujαj^2 - lj2/αjLj^2 

    Ujβj = U[j] - βj
    βjLj = βj - L[j]
    ljβj = lj1/Ujβj^2 - lj2/βjLj^2 

    fpj = sqrt(lj1)
    fqj = sqrt(lj2)
    xj = (fpj * Lj + fqj * Uj) / (fpj + fqj)
    xj = ifelse(ljαj >= 0, αj, ifelse(ljβj <= 0, βj, xj))

    return xj
end

# Primal problem functions
struct ApproxGradientUpdater{T, TV, TPD<:PrimalData{T, TV}, TM<:MMAModel{T, TV}}
    pd::TPD
    m::TM
end
ApproxGradientUpdater(pd::TPD, m::TM) where {T, TV, TPD<:PrimalData{T, TV}, TM<:MMAModel{T, TV}} = ApproxGradientUpdater{T, TV, TPD, TM}(pd, m)

function (gu::ApproxGradientUpdater{T})() where T
    @unpack pd, m = gu
    @unpack f_val, g_val, r = pd
    n = dim(m)
    r0 = f_val[] - mapreduce(gu, +, T(0), 1:n)
    old_r = copy(r)
    map!((i)->(g_val[i] - mapreduce(gu, +, T(0), Base.Iterators.product(1:n, i:i))), 
        r, 1:length(constraints(m)))
    pd.r0[] = r0
end
function (gu::ApproxGradientUpdater{T})(j::Int) where T
    pd = gu.pd
    @unpack x, L, U, p0, q0, ∇f = pd
    xj = x[j]
    xjLj = xj - L[j]
    Ujxj = U[j] - xj
    ∇fj = ∇f[j]
    (p0j, q0j) = ifelse(∇fj > 0, (abs2(Ujxj)*∇fj, zero(T)), (zero(T), -abs2(xjLj)*∇fj))
    p0[j], q0[j] = p0j, q0j
    return p0[j]/Ujxj + q0[j]/xjLj
end
function (gu::ApproxGradientUpdater{T})(ji::Tuple) where T
    j, i = ji
    pd = gu.pd
    @unpack x, L, U, p, q, ∇g = pd
    xjLj = x[j] - L[j]
    Ujxj = U[j] - x[j]
    ∇gj = ∇g[j,i]
    (pji, qji) = ifelse(∇gj > 0, (abs2(Ujxj)*∇gj, zero(T)), (zero(T), -abs2(xjLj)*∇gj))
    p[j,i], q[j,i] = pji, qji
    return pji/Ujxj + qji/xjLj
end

struct VariableBoundsUpdater{T, TV, TPD<:PrimalData{T, TV}, TModel<:MMAModel{T, TV}}
    pd::TPD
    m::TModel
    μ::T
end
VariableBoundsUpdater(pd::TPD, m::TModel, μ::T) where {T, TV, TPD<:PrimalData{T, TV}, TModel<:MMAModel{T, TV}} = VariableBoundsUpdater{T, TV, TPD, TModel}(pd, m, μ)

function (bu::VariableBoundsUpdater{T, TV})() where {T, TV}
    @unpack pd, m = bu
    @unpack α, β = pd
    n = dim(m)
    s = StructOfArrays{NTuple{2,T}, 1, Tuple{TV,TV}}((α, β))
    map!(bu, s, 1:n)
end
function (bu::VariableBoundsUpdater{T})(j) where T
    @unpack m, pd, μ = bu
    @unpack x, L, U = pd
    αj = max(L[j] + μ * (x[j] - L[j]), min(m, j))
    βj = min(U[j] - μ * (U[j] - x[j]), max(m, j))
    return (αj, βj)
end

struct AsymptotesUpdater{T, TV<:AbstractVector{T}, TModel<:MMAModel{T,TV}}
    m::TModel
    L::TV
    U::TV
    x::TV
    x1::TV
    x2::TV
    s_init::T
    s_incr::T
    s_decr::T
end

struct InitialAsymptotesUpdater{T, TV, TModel<:MMAModel{T,TV}}
    m::TModel
    x::TV
    s_init::T
end
Initial(au::AsymptotesUpdater) = InitialAsymptotesUpdater(au.m, au.x, au.s_init)
function InitialAsymptotesUpdater(m::TModel, x::TV, s_init::T) where {T, TV<:AbstractVector{T}, TModel<:MMAModel{T, TV}}
    InitialAsymptotesUpdater{T, TV, TModel}(m, x, s_init)
end

function (au::AsymptotesUpdater{T, TV})(k::Iteration) where {T, TV}
    @unpack L, U, m = au
    s = StructOfArrays{NTuple{2,T}, 1, Tuple{TV,TV}}((L, U))
    if k.i == 1 || k.i == 2
        map!(Initial(au), s, 1:dim(m))
    else
        map!(au, s, 1:dim(m))
    end
end
# Update move limits
function (au::InitialAsymptotesUpdater)(j::Int)
    @unpack m, x, s_init = au
    d = s_init * (max(m, j) - min(m, j))
    xj = x[j]
    Lj, Uj = xj-d, xj+d
    return (Lj, Uj)
end
function (au::AsymptotesUpdater)(j::Int)
    @unpack x, x1, x2, s_incr, s_decr, L, U = au
    xj = x[j]
    x1j = x1[j]
    x2j = x2[j]
    x1jLj = x1j - L[j]
    Ujx1j = U[j] - x1j
    (Lj, Uj) = ifelse(xor(xj >= x1j, x1j >= x2j), 
        (xj - x1jLj * s_decr, xj + Ujx1j * s_decr),
        (xj - x1jLj * s_incr, xj + Ujx1j * s_incr))
    return (Lj, Uj)
end
