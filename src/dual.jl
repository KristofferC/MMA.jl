struct DualTermEvaluator{T, TV, TPD<:PrimalData{T,TV}}
    pd::TPD
end
DualTermEvaluator(pd::TPD, λ::TV) where {T, TV, TPD<:PrimalData{T,TV}} = DualTermEvaluator{T, TV, TPD}(pd, λ)
function (dte::DualTermEvaluator)(λ, ji::Tuple)
    j, i = ji
    @unpack pd = dte
    @unpack p, q, ρ, σ, x1, x = pd
    σj = σ[j]
    Lj, Uj = minus_plus(x1[j], σj)
    xj = x[j]
    Ujxj = Uj - xj
    xjLj = xj - Lj
    Δ = ρ[i]*σj/4
    return λ[i]*((p[j,i] + Δ)/Ujxj + (q[j,i] + Δ)/xjLj)
end
function (dte::DualTermEvaluator)(j::Int)
    pd = dte.pd
    @unpack p0, q0, σ, x1, x = pd
    σj = σ[j]
    Lj, Uj = minus_plus(x1[j], σj)
    return p0[j]/(Uj-x[j]) + q0[j]/(x[j]-Lj)
end
struct DualObjVal{T, TV, TPD<:PrimalData}
    pd::TPD
    dte::DualTermEvaluator{T, TV, TPD}
    x_updater::XUpdater{TPD}
end
DualObjVal(pd::TPD, λ::TV, x_updater::XUpdater{TPD}) where {T, TV, TPD<:PrimalData{T, TV}} = DualObjVal(pd, DualTermEvaluator{T, TV, TPD}(pd), x_updater)
function (dobj::DualObjVal{T, TV, TPD})(λ) where {T, TV, TPD<:PrimalData{T}}
    @unpack pd, dte = dobj
    @unpack p, r, r0 = pd
    # Updates x to the Lagrangian minimizer for the input λ
    dobj.x_updater(λ)
    nv, nc = size(p)
    φ = r0[] + dot(λ, r)
    φ += mapreduce(dte, +, T(0), 1:nv)
    φ += mapreduce((ji)->dte(λ, ji), +, T(0), Base.Iterators.product(1:nv, 1:nc))
    return -φ
end

struct DualGradTermEvaluator{TPD<:PrimalData}
    pd::TPD
end
function (gte::DualGradTermEvaluator)(ji::Tuple)
    j, i = ji
    pd = gte.pd
    @unpack p, q, ρ, σ, x1, x = pd
    σj = σ[j]
    Lj, Uj = minus_plus(x1[j], σj)
    Ujxj = Uj - x[j]
    xjLj = x[j] - Lj
    Δ = ρ[i]*σj/4
    return (p[j,i] + Δ)/Ujxj + (q[j,i] + Δ)/xjLj
end
struct DualObjGrad{TPD<:PrimalData}
    pd::TPD
    gte::DualGradTermEvaluator{TPD}
    x_updater::XUpdater{TPD}
end
DualObjGrad(pd::PrimalData, x_updater::XUpdater) = DualObjGrad(pd, DualGradTermEvaluator(pd), x_updater)
function (dgrad::DualObjGrad{TPD})(∇φ::AbstractVector{T}, λ) where {T, TPD<:PrimalData{T}}
    @unpack pd, gte = dgrad
    @unpack p, r, r0 = pd
    # Updates x to the Lagrangian minimizer for the input λ
    dgrad.x_updater(λ)
    nv, nc = size(p)
    # Negate since we have a maximization problem
    map!((i)->(-r[i] - mapreduce(gte, +, T(0), Base.Iterators.product(1:nv, i:i))),
        ∇φ, 1:nc)    
    return ∇φ
end
