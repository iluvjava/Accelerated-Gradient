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

function perform_one_experiment()

end


function repeat_experiments_for()
end