include("../src/abstract_types.jl")
include("../src/non_smooth_fxns.jl")
include("../src/smooth_fxn.jl")
include("../src/proximal_gradient.jl")


using Test, LinearAlgebra, Plots, SparseArrays, LaTeXStrings


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
# x0 = LinRange(0, L, N) |> collect
x0 = ones(N)

MaxItr = 10000
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
results2 = rwapg(
    f, 
    g, 
    x0, 
    L, 
    L/2, 
    lipschitz_line_search=true, 
    estimate_scnvx_const=true,
    tol=tol, 
    max_itr=MaxItr
)

report_results(results1)
report_results(results2)

fxnVal1 = objectives(results1)
fxnVal2 = objectives(results2)

fxnMin = 0 # we already know the fmin. 

optimalityGap1 = @. fxnVal1 - fxnMin
optimalityGap1 = replace((x) -> max(x, eps(Float64)), optimalityGap1)

optimalityGap2 = @. fxnVal2 - fxnMin
optimalityGap2 = replace((x) -> max(x, eps(Float64)), optimalityGap2)


fig1 = plot(
    optimalityGap1, 
    label="V-FISTA",
    title="Simple Regression (N=$N)", 
    yaxis=:log10, 
    size=(600, 400), 
    linewidth=2, 
    ylabel="\n"*L"F(x_k) - F^*", 
    xlabel=L"k", 
    dpi=300
)
plot!(fig1, optimalityGap2, label="R-WAPG", linewidth=3)
fig1 |> display

savefig(fig1, "simple_regression_loss_$N.png")

muEstimates = results2.misc
fig2 = plot(
    muEstimates, 
    yaxis=:log10, 
    title="\n"*L"Simple Regression $μ_k$ Estimates", 
    size=(600, 400), 
    linewidth=2, 
    legend=:none,
    ylabel=L"μ_k", 
    xlabel=L"k", 
    dpi=300
)
display(fig2)
savefig(fig2, "simple_regression_loss_sc_estimates_$N.png")

