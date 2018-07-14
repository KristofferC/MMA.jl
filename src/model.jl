struct MMAModel{T, TV<:AbstractVector{T}, TC<:AbstractVector{<:Function}}
    dim::Int
    objective::Function
    ineq_constraints::TC
    box_max::TV
    box_min::TV
    # Trace flags
    store_trace::Bool
    show_trace::Bool
    extended_trace::Bool
    # Stopping criteria
    max_iters::Int
    ftol::Base.RefValue{T}
    xtol::Base.RefValue{T}
    grtol::Base.RefValue{T}
end

dim(m::MMAModel) = m.dim
min(m::MMAModel, i::Integer) = m.box_min[i]
max(m::MMAModel, i::Integer) = m.box_max[i]
min(m::MMAModel)= m.box_max
max(m::MMAModel) = m.box_min
objective(m::MMAModel) = m.objective
constraints(m::MMAModel) = m.ineq_constraints
constraint(m::MMAModel, i::Integer) = m.ineq_constraints[i]
eval_objective(m, g0::AbstractVector{T}, ∇g0) where {T} = T(m.objective(g0, ∇g0))
eval_constraint(m, i, g0::AbstractVector{T}, ∇g0) where {T} = T(constraint(m, i)(g0, ∇g0))
ftol(m) = m.ftol[]
xtol(m) = m.xtol[]
grtol(m) = m.grtol[]
ftol!(m, v) = m.ftol[] = v
xtol!(m, v) = m.xtol[] = v
grtol!(m, v) = m.grtol[] = v

MMAModel(args...; kwargs...) = MMAModel{Float64, Vector{Float64}, Vector{Function}}(args...; kwargs...)
function MMAModel{T, TV, TC}(dim,
                  objective::Function;
                  max_iters = 200,
                  xtol = T(eps(T)),
                  ftol = sqrt(T(eps(T))),
                  grtol = sqrt(T(eps(T))),
                  store_trace::Bool = false,
                  show_trace::Bool = false,
                  extended_trace::Bool = false) where {T, TV, TC}

    mins = ninfsof(TV, dim)
    maxs = infsof(TV, dim)
    MMAModel{T, TV, TC}(dim, objective, Function[],
             mins, maxs, store_trace, show_trace, extended_trace,
             max_iters, Ref(T(ftol)), Ref(T(xtol)), Ref(T(grtol)))
end

# Box constraints
function box!(m::MMAModel, i::Integer, minb::T, maxb::T) where {T}
    if !(1 <= i <= dim(m))
        throw(ArgumentError("box constraint need to applied to an existing variable"))
    end
    m.box_min[i] = minb
    m.box_max[i] = maxb
end

function box!(m::MMAModel, minb::T, maxb::T) where {T}
    nv = dim(m)
    map!((i)->minb, m.box_min, 1:nv)
    map!((i)->maxb, m.box_max, 1:nv)
end

function box!(m::MMAModel, minbs::Vector{T}, maxbs::Vector{T}) where {T}
    if (length(minbs) != dim(m)) || (length(minbs) != dim(m))
        throw(ArgumentError("box constraint vector must have same size as problem dimension"))
    end
    nv = dim(m)
    map!(identity, m.box_min, minbs)
    map!(identity, m.box_max, maxbs)
end

function ineq_constraint!(m::MMAModel, f::Function)
    push!(m.ineq_constraints, f)
end

function ineq_constraint!(m::MMAModel, fs::Vector{Function})
    for f in fs
        push!(m.ineq_constraints, f)
    end
end
