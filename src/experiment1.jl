include("abstract_types.jl")
include("non_smooth_fxns.jl")
include("smooth_fxn.jl")
include("proximal_gradient.jl")


using Test, LinearAlgebra, Plots, SparseArrays
# pgfplotsx()

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

N, μ, L = 1024, 1e-8, 1
f, g = make_quadratic_problem(N, μ, L)
# x0 = LinRange(0, L, N) |> collect
x0 = randn(N)

MaxItr = 5000
tol = 1e-8

results1 = vfista(
    f, 
    g, 
    x0, 
    L, 
    μ, 
    tol=tol, 
    max_itr=MaxItr
)

# results1 = fista(f, g, x0, tol=tol, max_itr=MaxItr)

results2 = inexact_vfista(
    f, 
    g, 
    x0, 
    lipschitz_constant=L, 
    sc_constant=L, 
    lipschitz_line_search=true, 
    sc_constant_line_search=true,
    tol=tol, 
    max_itr=MaxItr
)

report_results(results1)
report_results(results2)

fxnVal1 = get_all_objective_vals(results1)
fxnVal2 = get_all_objective_vals(results2)
# fxnMin = min(minimum(fxnVal1), minimum(fxnVal2))
fxnMin = 0

optimalityGap1 = @. fxnVal1 - fxnMin
optimalityGap1 = replace((x) -> max(x, eps(Float64)), optimalityGap1)

optimalityGap2 = @. fxnVal2 - fxnMin
optimalityGap2 = replace((x) -> max(x, eps(Float64)), optimalityGap2)


fig1 = plot(
    optimalityGap1, 
    label="v-fista",
    title="Simple regression", 
    yaxis=:log10, 
    size=(1200, 800)
)
plot!(fig1, optimalityGap2, label="inexact_vfista")
fig1 |> display
savefig(fig1, "simple_regression_loss.png")

muEstimates = results2.misc
fig2 = plot(
    muEstimates, 
    yaxis=:log10, 
    title="Simple regression Strong convexity index estimation, log_10", 
    size=(1200, 800)
)
display(fig2)
savefig(fig2, "simple_regression_loss_sc_estimates.png")
