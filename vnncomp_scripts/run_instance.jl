
using DPNeurifyFV
const DP = DPNeurifyFV



function write_counterexample(x, y, out_file)
    open(out_file, "w") do f
        for (i, xᵢ) in enumerate(x)
            if i == 1
                println(f, "((X_", i, " ", xᵢ, ")")
            else
                println(f, " (X_", i, " ", xᵢ, ")")
            end
        end

        for (i, yᵢ) in enumerate(y)
            if i == length(y)
                println(f, " (Y_", i, " ", yᵢ, "))")
            else
                println(f, " (Y_", i, " ", yᵢ, ")")
            end
        end
    end
end


function verify_an_instance(onnx_file, vnnlib_file, timeout)
    params = DP.PriorityOptimizerParameters(max_steps=999999999, print_frequency=100, timeout=timeout, stop_frequency=1, verbosity=2)
    solver = DPNFV(method=:DeepPolyRelax, max_vars=10)

    x_star, y_star, all_steps, result = DP.verify_vnnlib(solver, onnx_file, vnnlib_file, params, printing=true, check_onnx=true)

    if result == "SAT"
        return "sat", x_star, y_star
    elseif result == "UNSAT"
        return "unsat", x_star, y_star
    else
        return "unknown", x_star, y_star
    end
end


function main(args)
    onnx_file = args[1]
    vnnlib_file = args[2]
    out_file = args[3]
    timeout = parse(Int64, args[4])
    result, x_star, y_star = verify_an_instance(onnx_file, vnnlib_file, timeout)
    open(out_file, "w") do io
        write(io, result)

        if result == "sat"
            write_counterexample(x_star, y_star, out_file)
        end
    end
end

main(ARGS)