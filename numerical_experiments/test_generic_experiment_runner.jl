# This file is an example at how to use the generic experiment runner 
# to execute experimenr for a list of different algorithms. 


include("../numerical_experiments/generic_experiment_runner.jl")
using Test


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


@testset "TESTING THE GENERIC EXPERIMENT RUNNER. " begin

    function SanityCheck()
        return true
    end
    @test SanityCheck()
    # Prepare problem  parameters 
    tol = 1e-10
    max_itr = 5000
    N, μ, L = 512, 1e-5, 1
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
        L, 
        L/2, 
        lipschitz_line_search=true, 
        estimate_scnvx_const=true,
        tol=tol, 
        max_itr=max_itr
    )
    Algos = [VFISTA, RWAPG]
    function RunExperiments()
        global ExperimentResults = repeat_experiments_for(
            InitialGuessGuesser, Algos; repeat=3
        )
        return true
    end
    
    function VisualizeResults()
        # Get the fxn vals and gradient mapping for each algorithm. 
        global ExperimentResultsObjs = [ExperimentResults[k][1] for k in 1:length(ExperimentResults)]
        global ExperimentResultsGm = [ExperimentResults[k][2] for k in 1:length(ExperimentResults)]
        # Plot out the statistical results of the 2 algorithms. 

        return true
    end
    
    @test RunExperiments()
    @test VisualizeResults()




end