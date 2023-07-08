using PackageCompiler

create_sysimage(["DPNeurifyFV", "PyVnnlib", "LazySets", "Flux", "NeuralVerification", "NeuralPriorityOptimizer"], sysimage_path="sys_dpneurifyfv.so", precompile_execution_file=string(@__DIR__,"/precompile_template.jl"))