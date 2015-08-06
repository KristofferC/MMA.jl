function check_error(m, x0)
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
        if g(x0, []) > 0
            throw(ArgumentError("initial variable outside inequality constraint"))
        end
    end
end


