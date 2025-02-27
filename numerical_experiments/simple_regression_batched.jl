
include("../numerical_experiments/generic_experiment_runner.jl")
using Test, LaTeXStrings


function make_quadratic_problem(
    N::Integer,
    μ::Number, 
    L::Number
)::Tuple{SmoothFxn, NonsmoothFxn}
    diagonals = vcat([0], LinRange(μ, L, N - 1))
    A = diagm(diagonals) |> sparse
    b = zeros(N)
    f = Quadratic(A, b, 0)
    g = MAbs(0.0)
    return f, g
end


# Prepare problem  parameters 
tol = 1e-10
max_itr = 10000
N, μ, L = 1024, 1e-5, 1
f, g = make_quadratic_problem(N, μ, L)
InitialGuessGuesser = () -> randn(N)
# Package algorithm as runnables for testing. 
VFISTA = (x) -> vfista(
    f, 
    g, 
    x, 
    L, 
    μ, 
    tol=tol, 
    max_itr=max_itr
)
RWAPG = (x) -> rwapg(
    f, 
    g, 
    x,
    lipschitz_line_search=true, 
    estimate_scnvx_const=true,
    tol=tol, 
    max_itr=max_itr
)
MFISTA = (x) -> fista(
    f, 
    g, 
    x, 
    lipschitz_line_search=true, 
    tol=tol, 
    max_itr=max_itr, 
    mono_restart=true
)

Algos = [VFISTA, MFISTA, RWAPG]

function RunExperiments()
    global ExperimentResults = repeat_experiments_for(
        InitialGuessGuesser, Algos; repeat=40, true_minimum=0
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
        ylabel="Min, Max, Medium of \$\\delta_k\$",
        xlabel=L"k",
        title="\$\\delta_k\$ Statistics of Batched simple regression", 
        line=(3, :dash), 
        dpi=330, 
        
    )
    Medians = [qstats[3] for qstats in ExperimentResultsObjs[2]].|>log2
    Low = [qstats[1] for qstats in ExperimentResultsObjs[2]].|>log2
    High = [qstats[5] for qstats in ExperimentResultsObjs[2]].|>log2
    plot!(
        fig1, 
        Medians, 
        ribbon=(Medians .- Low, High .- Medians), 
        label="M-FISTA", 
        line=(3, :dot), 
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
    savefig(fig1, "simple_regression_batched-$N.png")
    return nothing
end



RunExperiments()
VisualizeResults()