using LinearAlgebra, Plots, SparseArrays





mutable struct SolutionHolder 
    
    solns::Vector{Number}
    gradient_norm::Vector{Number}
    function_values::vector{Number}
    sc_constants::Vector{Number}

    function SolutionHolder()
        this = new()
        return this
    end
end


function perform_index_vfista(
    x0::Number, 
    tol::Number; 
    max_itr=1000, 
    sc_constant=2, 
    lipschitz_constant=3, 
    solution_holder::SolutionHolder = SolutionHolder(), 
    a::Number = 1
)
    solns = Vector{Number}()
    gradient_norms = Vector{Number}()
    function_values = Vector{Number}()
    sc_constants = Vector{Number}()
    
    L, μ = lipschitz_constant, sc_constant
    f(x) = x^2 + sqrt(a + x^2)  # function 
    df(x) = 2*x + x/sqrt(a + x^2)  # derivative
    Df(x, y) = f(x) - f(y) - df(y)*(x - y) # Bregman divergence 
    x, y = x0, y0

    push!(solns, x)
    push!(gradient_norms, df(x))
    push!(function_values, f(x))
    push!(sc_constants, μ)
    
    for k = 1:max_itr
        x⁺ = x - df(x)/L
        ρ = sqrt(L/μ)
        θ = (ρ - 1)/(ρ + 1)
        y⁺ = x⁺ - θ*(x⁺ - x)
        μ = min(μ, Df(y⁺, y)/(y⁺ - y)^2)
        if abs(x - x⁺) < tol
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


