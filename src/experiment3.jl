
include("abstract_types.jl")
include("non_smooth_fxns.jl")
include("smooth_fxn.jl")
include("proximal_gradient.jl")


using Test, LinearAlgebra, Plots, SparseArrays


function make_logistic_loss_problem(
    M::Integer, 
    N::Integer
)::Tuple{SmoothFxn, NonsmoothFxn}
    # diagonals = vcat(LinRange(μ, L, N))
    A = abs.(randn(M, N))
    b = @. (cos(π*(0:1:M - 1)) + 1)/2
    f = LogitLoss(A, b, 0.01)
    g = MAbs(0.0)
    return f, g
end

M, N,= 1024, 785

f, g = make_logistic_loss_problem(M, N)
x0 = ones(N)
MaxItr = 5000
tol = 1e-8
results1 = fista(
    f, 
    g, 
    x0, 
    tol=tol, 
    max_itr=MaxItr,
    lipschitz_constant=1
)

results2 = inexact_vfista(
    f, 
    g, 
    x0, 
    lipschitz_constant=1, 
    sc_constant=1/2, 
    lipschitz_line_search=true, 
    sc_constant_line_search=true,
    tol=tol, 
    max_itr=MaxItr
)

report_results(results1)
report_results(results2)

fxnVal1 = get_all_objective_vals(results1)
fxnVal2 = get_all_objective_vals(results2)
fxnMin = min(minimum(fxnVal1), minimum(fxnVal2))
# fxnMin = 0

optimalityGap1 = @. fxnVal1 - fxnMin
optimalityGap1 = replace((x) -> max(x, eps(Float64)), optimalityGap1)

optimalityGap2 = @. fxnVal2 - fxnMin
optimalityGap2 = replace((x) -> max(x, eps(Float64)), optimalityGap2)

validIndx1 = findall((x) -> (x > 0), optimalityGap1)
validIndx2 = findall((x) -> (x > 0), optimalityGap2)

fig1 = plot(
    validIndx1,
    optimalityGap1[validIndx1], 
    yaxis=:log2,
    label="fista",
    size=(1200, 800),
    title="Logit Regression Experiment", 
)

plot!(
    fig1, 
    validIndx2,
    optimalityGap2[validIndx2], 
    label="inexact_vfista"
)
fig1 |> display
savefig(fig1, "logistic_regression_loss.png")

muEstimates = results2.misc
validIndx = findall((x) -> x > 0, muEstimates)
fig2 = plot(
    validIndx,
    muEstimates[validIndx], 
    yaxis=:log10, 
    size=(1200, 800),
    title="Strong convexity index estimation, log_10"
)
display(fig2)
savefig(fig2, "sc_estimates_logistic_loss.png")

gradmapNorm = results1.gradient_mapping_norm