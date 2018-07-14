struct DualTermEvaluator{T, TV, TPD<:PrimalData{T,TV}}
    pd::TPD
end
DualTermEvaluator(pd::TPD, λ::TV) where {T, TV, TPD<:PrimalData{T,TV}} = DualTermEvaluator{T, TV, TPD}(pd, λ)
function (dte::DualTermEvaluator)(λ, ji::Tuple)
    j, i = ji
    @unpack pd = dte
    @unpack p0, q0, p, q, U, L, x = pd
    Ujxj = U[j] - x[j]
    xjLj = x[j] - L[j]
    return λ[i]*p[j,i]/Ujxj + λ[i]*q[j,i]/xjLj
end
function (dte::DualTermEvaluator)(j::Int)
    pd = dte.pd
    @unpack p0, q0, U, L, x = pd
    return p0[j]/(U[j]-x[j]) + q0[j]/(x[j]-L[j])
end
struct DualObjVal{T, TV, TPD<:PrimalData}
    pd::TPD
    dte::DualTermEvaluator{T, TV, TPD}
    x_updater::XUpdater{TPD}
end
DualObjVal(pd::TPD, λ::TV, x_updater::XUpdater{TPD}) where {T, TV, TPD<:PrimalData{T, TV}} = DualObjVal(pd, DualTermEvaluator{T, TV, TPD}(pd), x_updater)
function (dobj::DualObjVal{T, TV, TPD})(λ) where {T, TV, TPD<:PrimalData{T}}
    @unpack pd, dte = dobj
    #@unpack p, r, r0 = pd
    @unpack p, r, r0, p0, q0, p, q, U, x, L = pd
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
    @unpack p, q, U, L, x = pd
    Ujxj = U[j] - x[j]
    xjLj = x[j] - L[j]
    return p[j,i]/Ujxj + q[j,i]/xjLj
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
