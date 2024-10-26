
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

iterates, G = itr_make(mu=0.01, theta=1)
iterates_diff = iterates[2:end] - iterates[1:end-1]
rayleigh = [dot(x, G*x)/dot(x, x) for x in iterates_diff]
fig = plot(rayleigh)
plot(fig)

