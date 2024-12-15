
include("../src/abstract_types.jl")
include("../src/non_smooth_fxns.jl")
include("../src/smooth_fxn.jl")
include("../src/proximal_gradient.jl")


using Test, LinearAlgebra, Plots, SparseArrays


function make_logistic_loss_problem(
    M::Integer, 
    N::Integer
)::Tuple{SmoothFxn, NonsmoothFxn}
    # diagonals = vcat(LinRange(μ, L, N))
    A = abs.(randn(M, N))
    b = @. (cos(π*(0:1:M - 1)) + 1)/2
    f = LogitLoss(A, b, 1e-4)
    g = MAbs(0.01)
    return f, g
end

M, N,= 64, 128

f, g = make_logistic_loss_problem(M, N)
x0 = ones(N)
MaxItr = 50000
tol = 1e-8
results1 = fista(
    f, 
    g, 
    x0, 
    tol=tol, 
    max_itr=MaxItr,
    lipschitz_constant=1, 
    lipschitz_line_search=true, 
    mono_restart=true
)

results2 = rwapg(
    f, 
    g, 
    x0, 
    1, 
    1/2;
    lipschitz_line_search=true, 
    estimate_scnvx_const=true,
    tol=tol, 
    max_itr=MaxItr
)

report_results(results1)
report_results(results2)

fxnVal1 = objectives(results1)
fxnVal2 = objectives(results2)
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
    yaxis=:log10,
    label="M-FISTA",
    size=(1200, 800),
    title="Logit Regression Experiment", 
)

plot!(
    fig1, 
    validIndx2,
    optimalityGap2[validIndx2], 
    label="R-WAPG"
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