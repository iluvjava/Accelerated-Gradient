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
    * `0` Exited and the algorithm reached desired tolerance. 
    * `1` Maximum iteration reached and then exit is triggered. 
- `collection_interval::Int`
    * Collect one solution per `collection_interval` iteration. 
- `itr_counter::Int`: An inner counter in the struct the keep track of the number of iterations, it's for mapping the 
results and using with the dictionary data structure. 
"""
mutable struct ProxGradResults
    
    gradient_mappings::Dict{Int, Vector}        # collect sparsely 
    gradient_mapping_norm::Vector{Real}         # always collect
    objective_vals::Dict{Int, Real}             # collected sparsely
    solns::Dict{Int, Vector}                    # collect sparsely 
    soln::Vector                                # the final solution
    step_sizes::Vector{Real}                    # always collect. 
    flags::Int                                   # collect at last
    collection_interval::Int                    # specified on initializations. 
    itr_counter::Int                            # updated within this class. 
    
    """
        You have the option to specify how frequently you want the results to be collected, because 
        if we collect all the solutions all the time the program is going to be a memory hog! 
    """
    function ProxGradResults(collection_interval::Int=typemax(Int))
        this = new()
        this.gradient_mappings = Dict{Int, Vector}()
        this.gradient_mapping_norm = Vector{Real}()
        this.objective_vals = Dict{Int, Real}()
        this.solns = Dict{Int, Vector}()
        this.step_sizes = Vector{Real}()
        this.flags = 0
        this.itr_counter = -1
        this.collection_interval = collection_interval
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
function Initiate!(
    this::ProxGradResults, 
    x0::Vector,  
    step_size::Real, 
    obj_initial::Union{Real, Nothing}=nothing
)
    this.itr_counter = 0
    this.solns[this.itr_counter] = x0
    if !isnothing(obj_initial)
        this.objective_vals[this.itr_counter] = obj_initial
    end
    push!(this.step_sizes, step_size)
    return nothing
end



"""
During each iteration, we have the option to store the parameters when the algorithm is running. 
- `this::ProxGradResults`: This is the type that the function acts on. 
- `soln::Vector`: This is the solution vector at the current iteration of the algorithm. 
"""
function Register!(
    this::ProxGradResults, 
    soln::Vector, 
    step_size::Real,
    pgrad_map::Union{Vector, Nothing}=nothing, 
    obj::Union{Real, Nothing}=nothing, 
)
    if this.itr_counter == -1
        error("ProxGrad Results is called without initiation.")
    end
    this.itr_counter += 1
    k = this.itr_counter

    push!(this.step_sizes, step_size)
    push!(this.gradient_mapping_norm, norm(pgrad_map))

    if mod(k, this.collection_interval) == 0
        this.solns[k] = copy(soln)
        this.gradient_mappings[k] = copy(pgrad_map)
        this.objective_vals[k] = obj
    end
    return nothing
end


"""
    Get all the sparsely collected solutions as an array of vectors. 
"""
function GetAllSolns(this::ProxGradResults)
    result = Vector{Vector}()
    for k in sort(keys(this.solns)|>collect)
        push!(result, this.solns[k])
    end
    return result
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

function grad_map(
    f::SmoothFxn, 
    g::NonsmoothFxn, 
    eta::Number, 
    x::AbstractArray
)::AbstractArray
return x - prox_grad(f, g, eta, x) end


