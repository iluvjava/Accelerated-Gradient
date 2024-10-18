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
    
    diagonals = vcat(LinRange(μ, L, N))
    A = diagm(diagonals)
    b = 10*ones(N)
    f = Quadratic(A, b, 0)
    g = MAbs(0.00001)
    return f, g
end



N, μ, L = 64, 1e-4, 1
f, g = make_lasso_problem(N, μ, L)
x0 = zeros(N)
global results1
global results2

results1 = vfista(f, g, x0, L, μ, eps=1e-5)
results2 = ppm_apg(
    f, 
    g, 
    x0, 
    lipschitz_constant=L, 
    sc_constant=1/2, 
    lipschitz_line_search=true, 
    sc_constant_line_search=true,
    eps=1e-5
)
report_results(results1)
report_results(results2)

fxnVal1 = get_all_objective_vals(results1)
fxnVal2 = get_all_objective_vals(results2)
fxnMin = min(minimum(fxnVal1), minimum(fxnVal2)) - 1e-15


optimalityGap1 = @. fxnVal1 - fxnMin
optimalityGap2 = @. fxnVal2 - fxnMin

idxStop1 = findlast((x) -> x > 0, optimalityGap1)
idxStop2 = findlast((x) -> x > 0, optimalityGap2)
idxStop = min(idxStop1, idxStop2)

xs = 1:idxStop

fig1 = plot(
    optimalityGap1[1:idxStop1], 
    label="v-fista", 
    yaxis=:log2, 
    title="Lasso experiment")
plot!(fig1,  optimalityGap2[1:idxStop2], label="ppm apg")
display(fig1)

muEstimates = results2.misc
fig2 = plot(1:length(muEstimates), muEstimates, yaxis=:log10)
display(fig2)




