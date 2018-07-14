macro mmatrace()
    esc(quote
        if tracing
            dt = Dict()
            if m.extended_trace
                dt["x"] = copy(x)
                dt["g(x)"] = copy(∇f_x)
                dt["λ"] = copy(results.minimizer)
            end
            update!(tr,
                    k,
                    f_x,
                    gr_residual,
                    dt,
                    m.store_trace,
                    m.show_trace)
        end
    end)
end
