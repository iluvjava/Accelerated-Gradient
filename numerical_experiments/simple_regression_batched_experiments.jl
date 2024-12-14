include("../src/abstract_types.jl")
include("../src/non_smooth_fxns.jl")
include("../src/smooth_fxn.jl")
include("../src/proximal_gradient.jl")
using Test, LinearAlgebra, Plots, SparseArrays


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

N, μ, L = 1024, 1e-6, 1
f, g = make_quadratic_problem(N, μ, L)

"""
Perform one run of multiple algorithm of interests, collect all objective value of the functions 
for each algorithm: "V-FISTA with constant parameters", "Parameter Free R-WAPG"
"""
function perform_one_experiment_on(
    f::SmoothFxn, 
    g::NonsmoothFxn, 
    x0::AbstractArray, 
    mu::Real, 
    L::Real; 
    max_itr::Int=5000, tol=10-8
)::NTuple{2, Vector}
    μ = mu
    _result1 = vfista(
        f, 
        g, 
        x0, 
        L, 
        μ, 
        tol=tol, 
        max_itr=max_itr
    )
    _results2 = rwapg(
        f, 
        g, 
        x0, 
        L, 
        L/2, 
        lipschitz_line_search=true, 
        estimate_scnvx_const=true,
        tol=tol, 
        max_itr=max_itr
    )
    _fxnVal1 = objectives(_result1)
    _fxnVal2 = objectives(_results2)
    return _fxnVal1, _fxnVal2
end


"""
Repeat `perform_one_experiment_on` for different initial gusses and collect statistics on objective values 
of the function at each iteration of the algorithm and return the statistics information. 

## Positional Argument 
"""
function repeat_experiments_for(
    # positional argument:
    f::SmoothFxn, 
    g::NonsmoothFxn, 
    mu::Real, 
    L::Real, 
    repeat_for=100,
    initial_guess_generator::Base.Callable; 
    # named argument: 
    max_itr::Int=5000
)


    
end