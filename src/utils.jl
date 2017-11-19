function check_error(m, x0)
    if length(x0) != dim(m)
        throw(ArgumentError("initial variable must have same length as number of design variables"))
    end

    for j in 1:length(x0)
        # x is not in box
        if !(min(m, j) <= x0[j] <= max(m,j))
            throw(ArgumentError("initial variable at index $j outside box constraint"))
        end
    end
end

# From Optim.jl
function assess_convergence(x::Array,
                            x_previous::Array,
                            f_x::Real,
                            f_x_previous::Real,
                            gr::Array,
                            xtol::Real,
                            ftol::Real,
                            grtol::Real)
    x_converged, f_converged, gr_converged = false, false, false

    if Optim.maxdiff(x, x_previous) < xtol
        x_converged = true
    end

    # Absolute Tolerance
    # if abs(f_x - f_x_previous) < ftol
    # Relative Tolerance
    if abs(f_x - f_x_previous) / (abs(f_x) + ftol) < ftol
        f_converged = true
    end

    if norm(vec(gr), Inf) < grtol
        gr_converged = true
    end

    converged = x_converged || f_converged || gr_converged

    return x_converged, f_converged, gr_converged, converged
end
