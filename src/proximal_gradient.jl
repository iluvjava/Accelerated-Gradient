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
    this.itr_counter = 0
    this.solns[this.itr_counter] = x0
    
    if !isnothing(obj_initial) &&   # initial objective value get passed
        this.collect_fxn_val        # instance is asked to collect function value
        this.objective_vals[this.itr_counter] = obj_initial
    end

    push!(this.step_sizes, step_size)
    return nothing
end

function get_current_iterate(this::ResultsCollector) return this.itr_counter end

function give_fxnval_nxt_itr(this::ResultsCollector) 
    return mod(this.itr_counter + 1, this.collection_interval) == 0 && this.collect_fxn_val
end

function give_pgradmap_nxt_itr(this::ResultsCollector) 
    return mod(this.itr_counter + 1, this.collection_interval) == 0 && this.collect_grad_map
end

function period_nxt_itr(this::ResultsCollector)
    return mod(this.itr_counter + 1, this.collection_interval) == 0
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
        error("ProxGrad Results is called without initiation.")
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
function get_all_solns(this::ResultsCollector)
    result = Vector{Vector}()
    for k in sort(keys(this.solns)|>collect)
        push!(result, this.solns[k])
    end
    return result
end


function report_results(this::ResultsCollector)::Nothing
    collectFxnVals = this.collect_fxn_val ? "yes" : "no"
    collectGPM = this.collect_grad_map ? "yes" : "no"
    initialized = this.itr_counter == -1 ? "no" : "yes"

    resultString = 
    """
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



abstract type AbstractAlgorithmSettings

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
    tol=1e-10
)::Union{Tuple{Real, AbstractVector}, Nothing}
    q(y, η) = f(y) - f(x) - dot(grad(f, x), y - x) - 1/(2η)*dot(y - x, y - x)
    y = prox_grad(f, g, eta, x)
    while eta >= tol && q(y, eta) > 0 
        eta /= 2 
        y = prox_grad(f, g, eta, x)
    end
    if eta >= tol
        return eta, y
    end
    return nothing
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
    x0::AbstractArray, 
    result_collector::ResultsCollector=ResultsCollector(), 
    eps::Real=1e-18,
    max_itr::Int=2000, 
    eta::Real=1, 
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

        if lipschitz_line_search
            result = execute_lipz_line_search(f, g, eta, xPre)
            if isnothing(result)
                flag = 2
                break # <-- Line search failed. 
            end
            eta, x = result
        else
            x = prox_grad(f, g, eta, x)
        end
        
        fxn_val, pgradMapVec = nothing, eta^(-1)*(x - xPre)
        if norm(x - xPre) < eps
            break # <-- Tolerance reached. 
        end
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
    x0::AbstractArray, 
    result_collector::ResultsCollector=ResultsCollector(), 
    eps::Real=1e-18,
    max_itr::Int=2000, 
    eta::Real=1, 
    lipschitz_line_search::Bool=true
)

end

function vfista(
    f::SmoothFxn, 
    g::NonsmoothFxn, 
    x0::AbstractArray, 
    result_collector::ResultsCollector=ResultsCollector(), 
    eps::Real=1e-18,
    max_itr::Int=2000, 
    lipschitz_constant::Real=1, 
    sc_constant::real=1,
    lipschitz_line_search::Bool=true
)



end

function ppm_apg()
end