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

N, μ, L = 1024, 1e-5, 1
f, g = make_quadratic_problem(N, μ, L)
# x0 = LinRange(0, L, N) |> collect
x0 = ones(N)

MaxItr = 10000
tol = 1e-10

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

results3 = fista(
    f, 
    g, 
    x0, 
    lipschitz_line_search=true, 
    tol=tol, 
    max_itr=MaxItr, 
    mono_restart=true
)

report_results(results1)
report_results(results2)
report_results(results3)

fxnVal1 = objectives(results1)
fxnVal2 = objectives(results2)
fxnVal3 = objectives(results3)

fxnMin = 0 # we already know the fmin. 

optimalityGap1 = @. fxnVal1 - fxnMin
optimalityGap1 = replace((x) -> max(x, eps(Float64)), optimalityGap1)

optimalityGap2 = @. fxnVal2 - fxnMin
optimalityGap2 = replace((x) -> max(x, eps(Float64)), optimalityGap2)

optimalityGap3 = @. fxnVal3 - fxnMin
optimalityGap3 = replace((x) -> max(x, eps(Float64)), optimalityGap3)

### PLOTTING THE FIRST PLOT. 
fig1 = plot(
    optimalityGap1, 
    label="V-FISTA",
    title="Simple Regression (N=$N)", 
    yaxis=:log10, 
    size=(600, 400), 
    line=(3, :dot),
    ylabel="\n"*L"F(x_k) - F^*", 
    xlabel=L"k", 
    dpi=300,
)
plot!(
    fig1, optimalityGap2, label="R-WAPG", line=(3, :dash),
)
plot!(
    fig1, optimalityGap3, label="M-FISTA", linewidth=3
)
fig1 |> display

savefig(fig1, "simple_regression_loss_$N.png")

### PLOTTING THE SECOND PLOT
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

### PLOTTING THE THIRD PLOT
rwapgUpperBnd = results2.rwapg_upperbnd
fig3 = plot(
    rwapgUpperBnd, 
    label="R-WAPG Theoretical UpperBnd",
    title="Simple Regression R-WAPG Upper Bound (N=$N)", 
    yaxis=:log10, 
    size=(700, 400), 
    line=(3, :dot),
    ylabel="\n\n"*L"\prod_{i = 0}^{k - 1}\max\left(1 - \alpha_{i + 1}, \frac{\alpha_{k + 1}^2}{\alpha_k^2}\right)", 
    xlabel=L"k", 
    dpi=300,
)
fig3 |> display
savefig(fig3, "simple_regression_rwapg_upperbnd_$N.png")