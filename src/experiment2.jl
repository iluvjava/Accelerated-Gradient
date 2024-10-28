include("abstract_types.jl")
include("non_smooth_fxns.jl")
include("smooth_fxn.jl")
include("proximal_gradient.jl")


using Test, LinearAlgebra, Plots, SparseArrays, Random
# pgfplotsx()



function make_lasso_problem(
    N::Integer, seed::Integer=224
)::Tuple{SmoothFxn, NonsmoothFxn, Real, Real}
    Random.seed!(seed)
    A = randn(N, N)
    b = cos.((π/2)*(0:1:N - 1)) .+ 1e-4*rand(N)
    f = SquareNormResidual(A, b)
    g = MAbs(0.1)
    μ = 1/norm(inv(A*A'))
    L = norm(A*A')
    return f, g, μ, L
end

N = 16
f, g, μ, L = make_lasso_problem(N)
x0 = randn(N)
MaxItr = 5000
tol = 1e-12

results1 = vfista(
    f, 
    g, 
    x0, 
    L, 
    μ, 
    tol=tol, 
    max_itr=MaxItr
)

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

results3 = fista(
    f, 
    g, 
    x0, 
    tol=tol, 
    max_itr=MaxItr, 
    lipschitz_constant=L, 
    lipschitz_line_search=false
)

report_results(results1)
report_results(results2)
report_results(results3)

fxnVal1 = get_all_objective_vals(results1)
fxnVal2 = get_all_objective_vals(results2)
fxnVal3 = get_all_objective_vals(results3)
fxnMin = min(minimum(fxnVal1), minimum(fxnVal2), minimum(fxnVal3))
# fxnMin = 0

optimalityGap1 = @. fxnVal1 - fxnMin
optimalityGap1 = replace((x) -> max(x, eps(Float64)), optimalityGap1)

optimalityGap2 = @. fxnVal2 - fxnMin
optimalityGap2 = replace((x) -> max(x, eps(Float64)), optimalityGap2)

optimalityGap3 = @. fxnVal3 - fxnMin
optimalityGap3 = replace((x) -> max(x, eps(Float64)), optimalityGap3)

validIndx1 = findall((x) -> (x > 0), optimalityGap1)
validIndx2 = findall((x) -> (x > 0), optimalityGap2)
validIndx3 = findall((x) -> (x > 0), optimalityGap3)


fig1 = plot(
    validIndx1,
    optimalityGap1[validIndx1], 
    yaxis=:log2,
    label="v-fista",
    title="LASSO Experiment", 
    size=(1200, 800)
)

plot!(
    fig1,
    validIndx3,
    optimalityGap3[validIndx3], 
    label="fista",
)

plot!(
    fig1, 
    validIndx2,
    optimalityGap2[validIndx2], 
    label="inexact_vfista"
)

fig1 |> display
savefig(fig1, "lasso_loss.png")


muEstimates = results2.misc
validIndx = findall((x) -> x > 0, muEstimates)
fig2 = plot(
    validIndx,
    muEstimates[validIndx], 
    yaxis=:log2, 
    size=(1200, 800),
    title="Lassso Strong convexity index estimation, log_10"
)
fig2 |> display
savefig(fig2, "lasso_sc_estimates.png")


