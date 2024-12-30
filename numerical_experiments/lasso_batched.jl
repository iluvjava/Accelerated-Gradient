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
        dpi=300,
        size=(600, 400)
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

