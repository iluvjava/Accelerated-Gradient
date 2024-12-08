include("../src/abstract_types.jl")
include("../src/non_smooth_fxns.jl")
include("../src/smooth_fxn.jl")
include("../src/proximal_gradient.jl")


using Test, LinearAlgebra, Plots, SparseArrays, Random
# pgfplotsx()



function make_lasso_problem(
    N::Integer, seed::Integer=224
)::Tuple{SmoothFxn, NonsmoothFxn, Real, Real}
    Random.seed!(seed)
    A = randn(N, N)
    x⁺ = cos.((π/2)*(0:1:N - 1)) .+ 1e-4*rand(N)
    b = A*x⁺
    f = SquareNormResidual(A, b)
    g = MAbs(0.1)
    μ = 1/norm(inv(A'*A))
    L = norm(A'*A)
    return f, g, μ, L
end

N = 64
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
    yaxis=:log10,
    label="v-fista",
    title="LASSO N=$N", 
    size=(600, 400), 
    linewidth=3, 
    dpi=300, 
    ylabel="Optimality Gap", 
    xlabel="Iteration"
)
plot!(
    fig1,
    validIndx3,
    optimalityGap3[validIndx3], 
    label="fista",
    linewidth=3, 
)
plot!(
    fig1, 
    validIndx2,
    optimalityGap2[validIndx2], 
    label="inexact_vfista", 
    linewidth=3
)

fig1 |> display
savefig(fig1, "lasso_loss_$N.png")


muEstimates = results2.misc
validIndx = findall((x) -> x > 0, muEstimates)
fig2 = plot(
    validIndx,
    muEstimates[validIndx], 
    yaxis=:log10, 
    size=(600, 400),
    title="μ_k Estimteas N=$N",
    ylabel="μ_k", 
    xlabel="Iteration", 
    linewidth=3
)
fig2 |> display
savefig(fig2, "lasso_sc_estimates_$N.png")


