include("abstract_types.jl")
include("non_smooth_fxns.jl")
include("smooth_fxn.jl")
include("proximal_gradient.jl")


using Test, LinearAlgebra, Plots


function make_lasso_problem(
    N::Integer,
    μ::Number, 
    L::Number
)::Tuple{SmoothFxn, NonsmoothFxn}
    diagonals = vcat([0], LinRange(μ, L, N - 1))
    A = diagm(diagonals)
    b = zeros(N)
    f = Quadratic(A, b, 0)
    g = MAbs(0.0)
    return f, g
end

N, μ, L = 32, 1e-2, 1
f, g = make_lasso_problem(N, μ, L)
x0 = ones(N)
MaxItr = 5000
tol = 1e-8
results1 = vfista(
    f, 
    g, 
    x0, 
    L, 
    μ, 
    eps=tol, 
    max_itr=MaxItr
)
results2 = inexact_vfista(
    f, 
    g, 
    x0, 
    lipschitz_constant=L/2, 
    sc_constant=1/5, 
    lipschitz_line_search=true, 
    sc_constant_line_search=true,
    eps=tol, 
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
    yaxis=:log2,
    label="v-fista",
    title="Simple regression", 
)
plot!(fig1, optimalityGap2, label="inexact_vfista", yaxis=:log2)
fig1 |> display



muEstimates = results2.misc
fig2 = plot(
    muEstimates, 
    yaxis=:log10, 
    title="Strong convexity index estimation"
)
display(fig2)
