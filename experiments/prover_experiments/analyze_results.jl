
using DataFrames, CSV

df = DataFrame(CSV.File(joinpath(@__DIR__, "./results/prover_mnist_results.csv")))
df_zono = DataFrame(CSV.File(joinpath(@__DIR__, "./results/prover_mnist_results_zono.csv")))

proven_correct = sum(df.upper .<= 0)
proven_incorrect = sum(df.lower .> 0)
# when center was incorrect, we didn't write to the df
center_incorrect = 1000 - size(df, 1)
unknown = 1000 - proven_correct - proven_incorrect - center_incorrect

println("verified ", proven_correct + proven_incorrect + center_incorrect, "/1000 (", proven_correct, " correct, ", 
        proven_incorrect, " incorrect, ", center_incorrect, " center incorrect)")

# PROVER only ran experiments on the first 100 correctly classified samples 
# TODO: are they the same images as in PyTorch MNIST? (i.e. same order in test set)
proven_correct_100 = sum(df.upper[1:100] .<= 0)
proven_incorrect_100 = sum(df.lower[1:100] .> 0)
unknown_100 = 100 - proven_correct_100 - proven_incorrect_100
println("verified ", proven_correct_100 + proven_incorrect_100, "/100 (", proven_correct_100, " correct, ", proven_incorrect_100, " incorrect)")


println("\n## Using Zonotope Domain")
proven_correct = sum(df_zono.upper .<= 0)
proven_incorrect = sum(df_zono.lower .> 0)
# when center was incorrect, we didn't write to the df
center_incorrect = 100 - size(df_zono, 1)
unknown = 100 - proven_correct - proven_incorrect - center_incorrect

println("verified ", proven_correct + proven_incorrect + center_incorrect, "/100 (", proven_correct, " correct, ", 
        proven_incorrect, " incorrect, ", center_incorrect, " center incorrect)")