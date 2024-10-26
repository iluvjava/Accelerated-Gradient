"""
This is a struct that models the results obtain from an execution of the 
prox gradient method. We see to store the following results and each result
is mapped to the iteration number of the algorithm.
## Fields

- `gradient_mappings::Dict{Int, Vector}`: gradient mapping is the difference between the solutions from the current 
iterations and the previous iterations. It's collect sparsely according the policy parameter *collection_interval*. 
- `gradient_mapping_norm::Vector{Real}`: The norm of the gradient mapping vector. 
- `objective_vals::Vector{Real}`: The objective values of the cost function. 
- `solns::Dict{Int, Vector}`: the solutions stored together with the indices. 
- `soln::Vector`: The final solution obtained by the algorithm. 
- `step_sizes::Vector{Real}`: all the stepsizes used by the algorithm. 
- `flags::Int`: exit flag
    * `-1` flag is not yet set by the algorithm 
    * `0` Exits because tolerance reached. 
    * `1` Maximum iteration reached without reaching the tolerance. 
    * `2` Lipschitz smoothness line search failed. 
- `collection_interval::Int`
    * Collect one solution per `collection_interval` iteration. 
- `itr_counter::Int`: An inner counter in the struct the keep track of the number of iterations, it's for mapping the 
results and using with the dictionary data structure. 
"""
mutable struct ResultsCollector
    

    gradient_mappings::Dict{Int, AbstractArray} # collect sparsely 
    gradient_mapping_norm::Vector{Real}         # always collect
    objective_vals::Dict{Int, Real}             # collected sparsely
    solns::Dict{Int, AbstractArray}             # collect sparsely 
    soln::AbstractArray                         # the solution of current iteration. 
    step_sizes::Vector{Real}                    # always collect. 
    
    flag::Int                                    # Status of terminations of the algorithm. 
    collection_interval::Int                    # specified on initializations. 
    itr_counter::Int                            # keep track of the iteration counter every call to the instance. 

    collect_fxn_val::Bool                       # whether to collect objective values periodically. 
    collect_grad_map::Bool                      # whether to collect gradient mapping periodically. 

    misc::Any                                   # whatever the client side wants, Initialized to be undefined. 

    
    """
        You have the option to specify how frequently you want the results to be collected, because 
        if we collect all the solutions all the time the program is going to be a memory hog! 
    """
    function ResultsCollector(
        collection_interval::Int=1,
        collect_fxn_val::Bool=true, 
        collect_grad_map::Bool=true, 
    )
        this = new()
        this.gradient_mappings = Dict{Int, AbstractArray}()
        this.gradient_mapping_norm = Vector{Real}()
        this.objective_vals = Dict{Int, Real}()
        this.solns = Dict{Int, AbstractArray}()
        this.step_sizes = Vector{Real}()
        this.flag = -1
        this.itr_counter = -1
        this.collection_interval = collection_interval

        this.collect_fxn_val = collect_fxn_val
        this.collect_grad_map = collect_grad_map
       return this
    end

end

"""
Initiate the instance of `ProxGradResults`, it's required, without this this class won't let you do anything. 
Because it at least need the initial conditions for the optimizations algorithm. 
### Argument
- `ProxGradResults`: The type that this function acts on. 
- `x0::Vecotr`: The initial guess for the algorithm. 
- `objective_initial::Real`: The initial objective value for the optimization problem. 
- `step_size::Real`: The initial step size for running the algorithm. 
"""
function initiate!(
    this::ResultsCollector, 
    x0::AbstractArray,  
    step_size::Real, 
    obj_initial::Union{Real, Nothing}=nothing
)
    if this.itr_counter != -1
        throw(ErrorException("Instance already been initiated, cannot be initiated for a second time. "))
    end

    this.itr_counter = 0
    this.solns[this.itr_counter] = x0
    
    if !isnothing(obj_initial) &&   # initial objective value get passed
        this.collect_fxn_val        # instance is asked to collect function value
        this.objective_vals[this.itr_counter] = obj_initial
    end

    push!(this.step_sizes, step_size)

    return nothing
end

function get_current_iterate(this::ResultsCollector)::Bool
return this.itr_counter end

function give_fxnval_nxt_itr(this::ResultsCollector)::Bool
    return mod(this.itr_counter + 1, this.collection_interval) == 0 && this.collect_fxn_val
end

function give_pgradmap_nxt_itr(this::ResultsCollector)::Bool
    return mod(this.itr_counter + 1, this.collection_interval) == 0 && this.collect_grad_map
end

function period_nxt_itr(this::ResultsCollector)::Bool
    return mod(this.itr_counter + 1, this.collection_interval) == 0
end

function collect_fxn_vals(this::ResultsCollector)::Bool
    return this.collect_fxn_val
end

function collect_grad_map(this::ResultsCollector)::Bool
    return this.collect_grad_map
end


"""
## What it this for: 
You must register results of current iterations if you want to increment the iteration counter of this struct. 
This is the function you call to do that. 
The setting of the instance determine what you should pass as results of current iteration into 
this function. 

- `this::ProxGradResults`: This is the type that the function acts on. 
- `soln::Vector`: This is the solution vector at the current iteration of the algorithm. 
- `step_size::Real`: This is the stepsize you used for the current iterations on the proximal gradient operator. 
- 
"""
function register!(
    this::ResultsCollector, 
    soln::AbstractArray, 
    step_size::Real,
    pgrad_map::Union{AbstractArray, Nothing}=nothing, 
    obj_val::Union{Real, Nothing}=nothing, 
)
    if this.itr_counter == -1
        throw(ErrorException("ProxGrad Results is called without initiation."))
    end
    
    # increment counter and register current solution. 
    this.itr_counter += 1
    k = this.itr_counter
    this.soln = copy(soln)

    push!(this.step_sizes, step_size)
    push!(this.gradient_mapping_norm, norm(pgrad_map))
    
    # periodically collect solutions, and updates pgrad_map, fxn_val, if asked. 
    if mod(k, this.collection_interval) == 0
        this.solns[k] = copy(soln)

        if this.collect_grad_map
            @assert !isnothing(pgrad_map) 
            this.gradient_mappings[k] = copy(pgrad_map)
        end

        if this.collect_fxn_val
            @assert !isnothing(obj_val) 
            this.objective_vals[k] = obj_val
        end
    end
    return nothing
end


"""
    Get all the sparsely collected solutions as an array of vectors. 
"""
function get_all_solns(this::ResultsCollector)::Vector{Vector}
    result = Vector{Vector}()
    for k in sort(keys(this.solns)|>collect)
        push!(result, this.solns[k])
    end
    return result
end

function get_all_objective_vals(this::ResultsCollector)::Vector{Number}
    result = Vector{Number}()
    for k in sort(keys(this.objective_vals)|>collect)
        push!(result, this.objective_vals[k])
    end
    return result
end



function report_results(this::ResultsCollector)::Nothing
    collectFxnVals = this.collect_fxn_val ? "yes" : "no"
    collectGPM = this.collect_grad_map ? "yes" : "no"
    initialized = this.itr_counter == -1 ? "no" : "yes"

    resultString = 
    """
    # RESULT REPORT FROM RESULT COLLECTOR 
    ## Collector Policies: 
        * Collect solution iterates periodically: $collectFxnVals
        * Collect gradient mapping periodically: $collectGPM
        * Collection period: $(this.collection_interval)
    ## Report of status
        * Initialized: $initialized
        * Current iteration counter: $(this.itr_counter)
        * exit flag: $(this.flag)
    
    """
    print(resultString)
    return nothing
end






# ==============================================================================
# PROXIMAL GRADIENT METHOD AND THEIR VARIANTS
# ==============================================================================

function prox_grad(
    f::SmoothFxn, 
    g::NonsmoothFxn, 
    eta::Number, 
    x::AbstractArray
)::AbstractArray
    @assert eta > 0 "Error, should be eta > 0 but we have eta = $(eta). "
    x = x - eta*grad(f, x)
    x = prox(g, eta, x)
    return x
end 


function pgrad_map(
    f::SmoothFxn, 
    g::NonsmoothFxn, 
    eta::Number, 
    x::AbstractArray
)::AbstractArray
return x - prox_grad(f, g, eta, x) end


function execute_lipz_line_search(
    f::SmoothFxn, 
    g::NonsmoothFxn,
    eta::Real, 
    x::AbstractVector,
    tol=1e-10; 
    # named vector
    lazy::Bool=false
)::Union{Tuple{Real, AbstractVector}, Nothing}
    
    d(y) = f(y) - f(x) - dot(grad(f, x), y - x) # <-- Bregman divergence of the smooth part. 
    y = prox_grad(f, g, eta, x)
    dotted = dot(y - x, y - x)
    
    if lazy
        # lazy then just update the Lipschitz constant, and not the future iterates. 
        eta = (1/2)*dotted/d(y)
    else
        while eta >= tol &&  d(y) >= 1/(2*eta)*dotted
            eta /= 2
            y = prox_grad(f, g, eta, x)
            dotted = dot(y - x, y - x)
        end
    end
    
    
    if eta >= tol
        return eta, y
    end

    return nothing
end

function execute_sc_const_line_search(
    f::SmoothFxn, 
    g::NonsmoothFxn, 
    lipschitz_constant::Real, 
    sc_constant::Real, 
    x::AbstractVector, 
    y::AbstractVector;
    # named arguments 
    tol=1e-10, 
    lazy::Bool=false
)::Union{Tuple{Real, AbstractVector}, Nothing}
    L = lipschitz_constant
    mu = sc_constant
    d(z) = f(z) - f(y) - dot(grad(f, y), z - y) # <-- Bregman divergence of the smooth part, fixed on x. 
    # d(z) = 2*dot(grad(f, z) - grad(f, y), z - y)
    Ty = prox_grad(f, g, 1/L, y)
    X(μ, x) = (1/sqrt(L/μ))*(y - x) + x - (1/sqrt(L/μ)/μ)*(y - Ty)  # <-- Probing the next "ghost" iterates based on the PPM APG interpretation 
    
    newX = X(mu, x)
    bregD = d(newX)
    dotted = dot(newX - x, newX - x)
    
    if lazy
        # only update mu
        mu = min(mu, 2*bregD/dotted)
    else
        # do a iterative line search using BD 
        while bregD < (mu/2)*dotted
            mu /= 2
            # mu = (2bregD/dotted)^2
            # update all because mu changed. 
            newX = X(mu, x)
            bregD = d(newX)
            dotted = dot(newX - x, newX - x)
        end
    end

    if mu >= tol 
        return mu, newX
    end
    # tolerance violated. 
    return mu, newX
end



"""
Madantory basic example to work through the implementations of the framework. 

## Positional Parameters
- `f`: an instance smooth function type. 
- `g`: an instance of non-smooth function type. 
- `x0`: An abstract array as the initial guess of the algorithm. 
- `result collector`: An instance of result collector for collecting the results 
and algorithm statistics. 
## Named parameters
- `eps`: pgrad map tolerance for terminations. 
- `max_itr`: The maximal iterations for terminations if it's reached. 
- `eta`: The initial stepsize for the algorithm. 
- `lipschitz_linear_search`: Wether to employ simple line search subroutine for 
determining the stepsize at each iteratin. 
"""
function ista(
    f::SmoothFxn, 
    g::NonsmoothFxn, 
    x0::AbstractArray;
    # named arguments 
    result_collector::ResultsCollector=ResultsCollector(),
    eps::Number=1e-18,
    max_itr::Int=2000, 
    eta::Number=1, 
    lipschitz_line_search::Bool=true
)::ResultsCollector

    # Initiates 
    x = x0
    flag = 0

    if give_fxnval_nxt_itr(result_collector)
        initialObjVal = f(x) + g(x)
    else
        initialObjVal = nothing
    end
    initiate!(result_collector, x0, eta,initialObjVal)
    
    # Iterates
    for k in 1:max_itr
        xPre = x

        # x is updated here 
        if lipschitz_line_search
            result = execute_lipz_line_search(f, g, eta, xPre)
            if isnothing(result) 
                flag = 2
                break # <-- Line fucking search failed. 
            end
            eta, x = result
        else
            x = prox_grad(f, g, eta, x)
        end
        
        fxn_val, pgradMapVec = nothing, eta^(-1)*(x - xPre)
        
        if give_fxnval_nxt_itr(result_collector)
            fxn_val = f(x) + g(x)
        end

        # collect results
        register!(
            result_collector, 
            x, 
            eta, 
            pgradMapVec, 
            fxn_val
        )

        if norm(x - xPre) < eps
            break # <-- Tolerance reached. 
        end

        if k == max_itr
            flag = 1 # <-- Maxmum iteration reached. 
        end
    end
    
    result_collector.flag = flag
    return result_collector
end



function fista(
    f::SmoothFxn, 
    g::NonsmoothFxn, 
    x0::AbstractArray;
    result_collector::ResultsCollector=ResultsCollector(),
    eps::Real=1e-18,
    max_itr::Int=2000, 
    eta::Real=1, 
    lipschitz_line_search::Bool=true
)

end


"""
The VIFISTA routine is a constant acceleration, constant stepsize method for function with quadratic growth 
parameter μ and Lipschitz constant on gradient L. 

## Poisitional Arguments
- `f`: 
- `g`: 
- `x0`: 
- `lipschitz_constant`: 
- `sc_constant`: 
- `result_collector`: 
- `eps`: 
- `max_itr`: 
"""
function vfista(
    f::SmoothFxn, 
    g::NonsmoothFxn, 
    x0::AbstractArray, 
    lipschitz_constant::Real,
    sc_constant::Real;
    # Named arguments 
    result_collector::ResultsCollector=ResultsCollector(), 
    eps::Number=1e-8,
    max_itr::Int=2000
)::ResultsCollector

    @assert sc_constant <= lipschitz_constant && sc_constant >= 0 "Expect `sc_constant` <= `lipschitz_constant`"+
    " and `sc_constant` = 0, however this is not true and we have: "+
    "`sc_constant`=$sc_constant, `lipschitz_constant`=$lipschitz_constant. "
    L, μ = lipschitz_constant, sc_constant
    κ = L/μ
    η = L^(-1)

    # initiate
    x, y = x0, x0
    fxnInitialVal = collect_fxn_vals(result_collector) ? f(x) + g(y) : nothing
    initiate!(result_collector, x0, η, fxnInitialVal)

    # iterates
    flag = 0
    for k = 1:max_itr
        
        x⁺ = prox_grad(f, g, η, y)
        y⁺ = x⁺ + ((sqrt(κ) - 1)/(sqrt(κ) + 1))*(x⁺ - x)
        
        # results collect
        fxn_val, pgradMapVec = nothing, η^(-1)*(x⁺ - x)
        if give_fxnval_nxt_itr(result_collector)
            fxn_val = f(x⁺) + g(x⁺)
        end
        # collect results
        register!(
            result_collector, 
            x⁺, 
            η, 
            pgradMapVec, 
            fxn_val
        )
        if norm(x⁺ - y) < eps
            break # <-- Tolerance reached. 
        else
            x = x⁺
            y = y⁺
        end

        if k == max_itr
            flag = 1
            break
        end
    end
    result_collector.flag = flag
    return result_collector
end


function fista(
    f::SmoothFxn, 
    g::NonsmoothFxn, 
    x0::AbstractArray, 
    lipschitz_constant::Real;
    # Named arguments 
    result_collector::ResultsCollector=ResultsCollector(), 
    eps::Number=1e-8,
    max_itr::Int=2000
)



end



function ppm_apg(
    f::SmoothFxn, 
    g::NonsmoothFxn, 
    x0::AbstractArray;
    # named arguments 
    result_collector::ResultsCollector=ResultsCollector(), 
    eps::Real=1e-8,
    max_itr::Int=2000, 
    lipschitz_constant::Real=1, 
    sc_constant::Real=0.5,
    lipschitz_line_search::Bool=false, 
    sc_constant_line_search::Bool=false
)::ResultsCollector
    throw(ErrorException("ALGORITHM DEPRECATED. "))
    @assert sc_constant <= lipschitz_constant && sc_constant >= 0 "Expect `sc_constant` <= `lipschitz_constant`"*
    " and `sc_constant` = 0, however this is not true and we have: "*
    "`sc_constant`=$sc_constant, `lipschitz_constant`=$lipschitz_constant. "
    L, μ = lipschitz_constant, sc_constant
    # initiate
    x, y = x0, x0
    flag = 0
    fxnInitialVal = collect_fxn_vals(result_collector) ? f(x) + g(y) : nothing
    initiate!(result_collector, x0, 1/L, fxnInitialVal)
    scConstEstimates = Vector{Number}()
    # iterates 
    for k in 1:max_itr
        if lipschitz_line_search
            results = execute_lipz_line_search(f, g, 1/L, y)
            if isnothing(results)
                flag = 2   
                break # <--  Lipschitz line search breaks near point: y
            end
            s, Ty = results 
            L = 1/s
        else
            Ty = prox_grad(f, g, 1/L, y)
        end
        if sc_constant_line_search
            results = execute_sc_const_line_search(f, g, L, μ, x, y)
            if isnothing(results)
                flag = 2 # <-- Strong sc constant search failed 
                break 
            end
            μ, x⁺ = results
            push!(scConstEstimates,  μ)
        else
            ρ = sqrt(L/μ)
            x⁺ = (1/ρ)*(y - x) + x - (1/(ρ*μ))*L*(y - Ty)
            #  (1/sqrt(L/μ))*(y - x) + x - (1/sqrt(L/μ)/μ)*(y - Ty)
        end
        ρ = sqrt(L/μ)
        y⁺ = (ρ*Ty + x⁺)/(1 + ρ)
        # results collect
        fxn_val, pgradMapVec = nothing, L*(y - Ty)
        if give_fxnval_nxt_itr(result_collector)
            fxn_val = f(Ty) + g(Ty)
        end
        register!(
            result_collector, 
            x, 
            1/L, 
            pgradMapVec, 
            fxn_val
        )
        if norm(y - Ty) < eps
            println("tolerance reached. ")
            break # <-- Tolerance reached. 
        else
            x, y = x⁺, y⁺
        end
        if k == max_itr
            flag = 1
            # max iteration reached
        end
    end
    result_collector.misc = scConstEstimates
    result_collector.flag = flag
    return result_collector
end



function inexact_vfista(
    f::SmoothFxn, 
    g::NonsmoothFxn, 
    x0::AbstractArray;
    lipschitz_constant::Real=1,
    sc_constant::Real=1/2, 
    # Named arguments 
    result_collector::ResultsCollector=ResultsCollector(), 
    eps::Number=1e-8,
    max_itr::Int=2000, 
    sc_constant_line_search::Bool=false, 
    lipschitz_line_search::Bool=false
)::ResultsCollector

    @assert sc_constant <= lipschitz_constant && sc_constant >= 0 "Expect `sc_constant` <= `lipschitz_constant`"+
    " and `sc_constant` = 0, however this is not true and we have: "+
    "`sc_constant`=$sc_constant, `lipschitz_constant`=$lipschitz_constant. "
    L, μ = lipschitz_constant, sc_constant
    κ = L/μ

    # initiate
    x, y = x0, x0 # current ieration of x, y
    ỹ = y # previous iteration of y 
    t = (n) -> (n + 1)/2
    fxnInitialVal = collect_fxn_vals(result_collector) ? f(x) + g(y) : nothing
    estimatedSCConst = Vector{Number}()
    push!(estimatedSCConst, μ)
    initiate!(result_collector, x0, 1/L, fxnInitialVal)

    # iterate
    flag = 0
    for k = 1:max_itr
        
        # make future iterature x⁺
        if lipschitz_line_search 
            results = execute_lipz_line_search(f, g, 1/L, y)
            if isnothing(results)
                flag = 2 
                break 
            end
            η, x⁺ = results 
            L = 1/η  # update Lipschitz constant and iterates
        else
            x⁺ = prox_grad(f, g, 1/L, y)
        end
        # estimate strong convexity constant
        if sc_constant_line_search  # estimate strongly convex constant
            Df(x1, x2) = f(x1) - f(x2) - dot(grad(f, x2), x1 - x2)
            μ = (2*Df(x, x⁺)/dot(x - x⁺, x - x⁺) + μ)/2
            @assert !isnan(μ)
        end
        κ = L/μ
        θ = μ > 0 ? ((sqrt(κ) - 1)/(sqrt(κ) + 1)) : 0
        y⁺ = x⁺ + θ*(x⁺ - x)
        push!(estimatedSCConst, μ)
        # results collect
        fxn_val, pgradMapVec = nothing, (1/L)*(x⁺ - x)
        if give_fxnval_nxt_itr(result_collector)
            fxn_val = f(x⁺) + g(x⁺)
        end
        # collect results
        register!(
            result_collector, 
            x⁺, 
            1/L, 
            pgradMapVec, 
            fxn_val
        )
        if norm(x⁺ - y) < eps
            break # <-- Tolerance reached. 
        else
            x = x⁺
            ỹ = y
            y = y⁺
        end
        
        if k == max_itr
            flag = 1
            break
        end
    end
    
    result_collector.flag = flag
    result_collector.misc = estimatedSCConst
    return result_collector
end