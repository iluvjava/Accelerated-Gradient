include("../src/abstract_types.jl")
include("../src/non_smooth_fxns.jl")
include("../src/smooth_fxn.jl")
include("../src/proximal_gradient.jl")
include("../numerical_experiments/generic_experiment_runner.jl")

using Test, LinearAlgebra, Plots, SparseArrays, Random
# pgfplotsx()

function make_lasso_problem(
    M::Integer,
    N::Integer, 
    seed::Integer=1
)::Tuple{SmoothFxn, NonsmoothFxn, Real, Real}
    Random.seed!(seed)
    A = randn(M, N)
    x⁺ = cos.((π/2)*(0:1:N - 1))
    b = A*x⁺
    f = SquareNormResidual(A, b)
    g = MAbs(0.3)
    μ = 1/norm(inv(A'*A))
    L = norm(A'*A)
    return f, g, μ, L
end

M, N = 64, 256
f, g, μ, L = make_lasso_problem(M, N)
x0 = randn(N)
MaxItr = 8000
tol = 1e-6
InitialGuessGuesser = () -> randn(N)
VFISTA = (x) -> vfista(
    f, 
    g, 
    x, 
    L, 
    μ, 
    tol=tol, 
    max_itr=MaxItr
)
RWAPG = (x) -> rwapg(
    f, 
    g, 
    x,
    lipschitz_line_search=true, 
    estimate_scnvx_const=true,
    tol=tol, 
    max_itr=MaxItr
)
MFISTA = (x) -> fista(
    f, 
    g, 
    x, 
    lipschitz_line_search=true, 
    tol=tol, 
    max_itr=MaxItr, 
    mono_restart=true
)
Algos = [VFISTA, MFISTA, RWAPG]

function RunExperiments()
    global ExperimentResults = repeat_experiments_for(
        InitialGuessGuesser, Algos; repeat=30
    )
    return nothing
end


function VisualizeResults()
    # Get the fxn vals and gradient mapping for each algorithm. 
    global ExperimentResultsObjs = [ExperimentResults[k][1] for k in 1:length(ExperimentResults)]
    global ExperimentResultsGm = [ExperimentResults[k][2] for k in 1:length(ExperimentResults)]
    # Plot out the statistical results of the 2 algorithms. 

    Medians = [qstats[3] for qstats in ExperimentResultsObjs[1]].|>log2
    Low = [qstats[1] for qstats in ExperimentResultsObjs[1]].|>log2
    High = [qstats[5] for qstats in ExperimentResultsObjs[1]].|>log2
    fig1 = plot(
        Medians, 
        ribbon=(Medians .- Low, High .- Medians),
        label="V-FISTA", 
        title="Batched LASSO Experiments Statistics",
        ylabel="Normalized Optimality Gap Stats",
        xlabel="Iteration Count",
        linewidth=3, 
        style=:dash, 
        size=(600, 400),
    )
    Medians = [qstats[3] for qstats in ExperimentResultsObjs[2]].|>log2
    Low = [qstats[1] for qstats in ExperimentResultsObjs[2]].|>log2.|> ((x) -> max(x, -54))
    High = [qstats[5] for qstats in ExperimentResultsObjs[2]].|>log2
    plot!(
        fig1, 
        Medians, 
        ribbon=(Medians .- Low, High .- Medians), 
        label="M-FISTA", 
        linewidth=3, 
        style=:dot
    )
    Medians = [qstats[3] for qstats in ExperimentResultsObjs[3]].|>log2
    Low = [qstats[1] for qstats in ExperimentResultsObjs[3]].|>log2
    High = [qstats[5] for qstats in ExperimentResultsObjs[3]].|>log2
    plot!(
        fig1, 
        Medians, 
        ribbon=(Medians .- Low, High .- Medians), 
        label="Free R-WAPG", 
        linewidth=3
    )
    fig1|>display
    savefig(fig1, "lasso_batched_statistics_$M-$N.png")
    return nothing
end


RunExperiments()
VisualizeResults()


# results1 = vfista(
#     f, 
#     g, 
#     x0, 
#     L, 
#     μ, 
#     tol=tol, 
#     max_itr=MaxItr
# )

# @info "VFISTA DONE"
# results2 = rwapg(
#     f, 
#     g, 
#     x0, 
#     lipschitz_line_search=true, 
#     estimate_scnvx_const=true,
#     tol=tol, 
#     max_itr=MaxItr
# )

# @info "R-WAPG DONE"
# results3 = fista(
#     f, 
#     g, 
#     x0, 
#     tol=tol, 
#     max_itr=MaxItr, 
#     lipschitz_constant=L, 
#     lipschitz_line_search=false, 
#     mono_restart=true, 
# )
# @info "M-FISTA DONE"

# report_results(results1)
# report_results(results2)
# report_results(results3)

# fxnVal1 = objectives(results1)
# fxnVal2 = objectives(results2)
# fxnVal3 = objectives(results3)
# fxnMin = min(minimum(fxnVal1), minimum(fxnVal2), minimum(fxnVal3))
# # fxnMin = 0

# optimalityGap1 = @. fxnVal1 - fxnMin
# optimalityGap1 = replace((x) -> max(x, eps(Float64)), optimalityGap1)
# optimalityGap2 = @. fxnVal2 - fxnMin
# optimalityGap2 = replace((x) -> max(x, eps(Float64)), optimalityGap2)
# optimalityGap3 = @. fxnVal3 - fxnMin
# optimalityGap3 = replace((x) -> max(x, eps(Float64)), optimalityGap3)

# validIndx1 = findall((x) -> (x > 0), optimalityGap1)
# validIndx2 = findall((x) -> (x > 0), optimalityGap2)
# validIndx3 = findall((x) -> (x > 0), optimalityGap3)


# fig1 = plot(
#     validIndx1,
#     optimalityGap1[validIndx1], 
#     yaxis=:log2,
#     label="V-FISTA",
#     title="LASSO N=$N", 
#     size=(600, 400), 
#     linewidth=3, 
#     dpi=300, 
#     ylabel="Optimality Gap", 
#     xlabel="Iteration"
# )
# plot!(
#     fig1,
#     validIndx3,
#     optimalityGap3[validIndx3], 
#     label="M-FISTA",
#     linewidth=3, 
# )
# plot!(
#     fig1, 
#     validIndx2,
#     optimalityGap2[validIndx2], 
#     label="FREE RWAPG", 
#     linewidth=3
# )

# fig1 |> display
# savefig(fig1, "lasso_loss_$N.png")


# muEstimates = results2.misc
# validIndx = findall((x) -> x > 0, muEstimates)
# fig2 = plot(
#     validIndx,
#     muEstimates[validIndx], 
#     yaxis=:log10, 
#     size=(600, 400),
#     title="μ_k Estimteas N=$N",
#     ylabel="μ_k", 
#     xlabel="Iteration", 
#     linewidth=3
# )
# fig2 |> display
# savefig(fig2, "lasso_sc_estimates_$N.png")
