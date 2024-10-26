using LinearAlgebra, Plots, SparseArrays



mutable struct SolutionHolder 
    
    solns::Vector{Number}
    gradient_norms::Vector{Number}
    function_values::Vector{Number}
    sc_constants::Vector{Number}

    function SolutionHolder()
        this = new()
        return this
    end
end


function perform_inexact_vfista(
    x0::Number; 
    tol::Number=1e-10,
    max_itr=1000, 
    sc_constant=nothing, 
    lipschitz_constant=nothing, 
    solution_holder::SolutionHolder = SolutionHolder(), 
    a::Number=0.1
)
    solns = Vector{Number}()
    gradient_norms = Vector{Number}()
    function_values = Vector{Number}()
    sc_constants = Vector{Number}()
    if isnothing(sc_constant)
        sc_constant = 2
    end
    if isnothing(lipschitz_constant)
        lipschitz_constant = 2 + 1/a
    end
    
    L, μ = lipschitz_constant, sc_constant
    f(x) = x^2 + sqrt(a + x^2)  # function 
    df(x) = 2*x + x/sqrt(a + x^2)  # derivative
    Df(x, y) = f(x) - f(y) - df(y)*(x - y) # Bregman divergence 
    x, y = x0, x0

    push!(solns, x)
    push!(gradient_norms, df(x))
    push!(function_values, f(x))
    push!(sc_constants, μ)
    
    for k = 1:max_itr
        x⁺ = y - df(y)/L
        ρ = sqrt(L/μ)
        θ = (ρ - 1)/(ρ + 1)
        y⁺ = x⁺ - θ*(x⁺ - x)
        μ = max(min(μ, 2*Df(y⁺, y)/(y⁺ - y)^2), 0)

        push!(solns, x)
        push!(gradient_norms, abs(x⁺ - y))
        push!(function_values, f(x))
        push!(sc_constants, μ)

        if abs(x⁺ - y) < tol
            break 
        end
        x = x⁺
        y = y⁺
    end
    
    solution_holder.solns = solns
    solution_holder.gradient_norms = gradient_norms
    solution_holder.function_values = function_values
    solution_holder.sc_constants = sc_constants

    return solution_holder
end


a = 0.1
L = 2 + 1/a

solution_holder = perform_inexact_vfista(
    0.1,
    sc_constant=L, 
    tol=1e-4
)
fxn_vals = solution_holder.function_values
sc_cosntants = solution_holder.sc_constants
iterations = 1:length(fxn_vals)

fig1 = plot(fxn_vals)
display(fig1)
fig2 = plot(iterations, sc_cosntants)
