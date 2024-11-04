
using Plots, LinearAlgebra, SparseArrays


function itr_make(
    ;
    mu::Number=0.1,
    L::Number=1,
    theta::Number=1, 
    N::Number=3,
    max_itr::Number=3000, 
    tol::Number=1e-4
)
    θ = theta
    y = zeros(N); y[end - 1] = 10
    x = copy(y)
    G = Diagonal(LinRange(mu, L, N)) 
    T = (x) -> x - G*x
    iterates = Vector{AbstractArray}()
    push!(iterates, y)
    

    for k = 1:max_itr
        x⁺ = T(y)
        y⁺ =  x⁺ + θ*(x⁺ - x)
        
        if norm(y⁺ - y) < tol
            break
        end

        push!(iterates, y⁺)
        y = y⁺
        x = x⁺
        
    end

    return iterates, G
end


"""
Simple 2x2 linear recurrence iteration matrix for analysis. 
"""
function simple_iter_matrix_make(
    ;tau=0.4, theta=1
)
    θ, τ = theta, tau
    M = [0 1; -θ*τ (1 + θ)*τ]
    return M
end

function full_iter_matrix_make()

end


# Changes in Rayleigh quotient. 
begin
    iterates, G = itr_make(mu=0.01, theta=1)
    iterates_diff = iterates[2:end] - iterates[1:end-1]
    rayleigh = [dot(x, G*x)/dot(x, x) for x in iterates_diff]
    fig = plot(rayleigh)
    plot(fig)    
end


# Spectral radius plotted against the momentum theta. 
begin
    thetas = LinRange(0, 1, 1000)
    τ = 1
    θ⁺ = (sqrt(1/(1 - τ)) - 1)/(sqrt(1/(1 - τ)) + 1)
    maxEigens = Vector()
    imaginary = Vector()
    for θ in thetas
        M = simple_iter_matrix_make(theta=θ, tau=τ)
        Λ, V = eigen(M)
        push!(maxEigens, Λ .|> abs |> maximum)
        push!(imaginary, imag(Λ[1]) |> abs)
    end
    fig = plot(
        thetas, 
        maxEigens, 
        label="ρ(M)", 
        legend=:bottomleft
    )
    plot!(fig, thetas, imaginary, label="IMAG(λ)", title="Spectral Radius: τ=$τ")
    vline!(fig, [θ⁺], label="θ⁺≈$(round(θ⁺, sigdigits=2))")
    display(fig)
end


# spectral radius 2d visualizations plot 
begin


end





