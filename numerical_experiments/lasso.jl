
include("../src/abstract_types.jl")
include("../src/non_smooth_fxns.jl")
include("../src/smooth_fxn.jl")
include("../src/proximal_gradient.jl")


using Test, LinearAlgebra, Plots, SparseArrays, Random
# pgfplotsx()

function make_lasso_problem(
    M::Integer,
    N::Integer, 
    seed::Integer=1
)::Tuple{SmoothFxn, NonsmoothFxn, Real, Real}
    Random.seed!(seed)
    A = randn(M, N)
    x⁺ = cos.((π/2)*(0:1:N - 1))
    b = A*x⁺
    f = SquareNormResidual(A, b)
    g = MAbs(0.3)
    μ = 1/norm(inv(A'*A))
    L = norm(A'*A)
    return f, g, μ, L
end

M, N = 64, 256
f, g, μ, L = make_lasso_problem(M, N)
println("Parameters out. μ = $μ, L = $L")
x0 = randn(N)
MaxItr = 8000
tol = 1e-6
InitialGuessGuesser = () -> randn(N)

results1 = vfista(
    f, 
    g, 
    x0, 
    L, 
    μ, 
    tol=tol, 
    max_itr=MaxItr
)

@info "VFISTA DONE"
results2 = rwapg(
    f, 
    g, 
    x0, 
    lipschitz_line_search=true, 
    estimate_scnvx_const=true,
    tol=tol, 
    max_itr=MaxItr
)

@info "R-WAPG DONE"
results3 = fista(
    f, 
    g, 
    x0, 
    tol=tol, 
    max_itr=MaxItr, 
    lipschitz_constant=L, 
    lipschitz_line_search=false, 
    mono_restart=true, 
)
@info "M-FISTA DONE"

report_results(results1)
report_results(results2)
report_results(results3)

fxnVal1 = objectives(results1)
fxnVal2 = objectives(results2)
fxnVal3 = objectives(results3)
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
    label="V-FISTA",
    title="LASSO Experiment Optimality Gap", 
    size=(600, 400), 
    line=(3, :dot), 
    dpi=300, 
    ylabel="\n"*L"F(x_k) - F^*", 
    xlabel="k", 
    legend=:bottomleft
)
plot!(
    fig1,
    validIndx3,
    optimalityGap3[validIndx3], 
    label="M-FISTA",
    line=(3, :dash), 
)
plot!(
    fig1, 
    validIndx2,
    optimalityGap2[validIndx2], 
    label="FREE RWAPG", 
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
    title="\n\$μ_k\$ Estimteas N=$N",
    ylabel=L"μ_k", 
    xlabel=L"k", 
    linewidth=3, 
    legend=:none
)
fig2 |> display
savefig(fig2, "lasso_sc_estimates_$N.png")
