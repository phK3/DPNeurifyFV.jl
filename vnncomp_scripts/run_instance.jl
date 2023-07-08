
using DPNeurifyFV
const DP = DPNeurifyFV


function verify_an_instance(onnx_file, vnnlib_file)
    params = DP.PriorityOptimizerParameters(max_steps=10, print_frequency=1, stop_frequency=1, verbosity=2)
    solver = DPNFV(method=:DeepPolyRelax)

    x_star, y_star, all_steps, result = DP.verify_vnnlib(solver, onnx_file, vnnlib_file, params, printing=true)

    if result == "SAT"
        return "sat", x_star, y_star
    elseif result == "UNSAT"
        return "unsat", x_star, y_star
    else
        return "unknown", x_star, y_star
    end
end


function main(args)
    result = verify_an_instance(args[1], args[2])
    open(args[3], "w") do io
        write(io, result[1])

        if result[1] == "sat"
            for (i, x) in enumerate(vec(x_star))
                if i == 0
                    write(io, "\n(")
                else
                    write(io, "\n")
                end

                write(io, "(X_$i $x)")
            end

            for (i, y) in enumerate(vec(y_star))
                write(io, "\n(Y_$i $y)")
            end

            write(io, ")")
        end
    end
end

main(ARGS)