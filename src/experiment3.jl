
include("abstract_types.jl")
include("non_smooth_fxns.jl")
include("smooth_fxn.jl")
include("proximal_gradient.jl")


using Test, LinearAlgebra, Plots, SparseArrays
pgfplotsx()


function make_lasso_problem(
    N::Integer,
    μ::Number, 
    L::Number
)::Tuple{SmoothFxn, NonsmoothFxn}
    diagonals = vcat(LinRange(μ, L, N))
    A = diagm(diagonals) |> sparse
    b = @. (cos(π*(0:1:N - 1)) + 1)/2
    f = LogitLoss(A, b)
    g = MAbs(1)
    return f, g
end

N, μ, L = 16, 1e-4, 1
f, g = make_lasso_problem(N, μ, L)
x0 = N:-1:1 |> collect
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
    lipschitz_constant=L, 
    sc_constant=L, 
    lipschitz_line_search=true, 
    sc_constant_line_search=true,
    eps=tol, 
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
    label="v-fista",
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

gradmapNorm = results1.gradient_mapping_norm