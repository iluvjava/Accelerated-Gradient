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
"""
function register!(
    this::ResultsCollector, 
    soln::AbstractArray, 
    step_size::Real,
    pgrad_map::Union{AbstractArray, Nothing}=nothing, 
    obj_val::Union{Real, Nothing}=nothing, 
)
    if this.itr_counter == -1
        throw(ErrorException(
            "ProxGrad Results is called without initiation."*
            "Please initiate instance via the `initiate!` function. "
            ))
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
function solns(this::ResultsCollector)::Vector{Vector}
    result = Vector{Vector}()
    for k in sort(keys(this.solns)|>collect)
        push!(result, this.solns[k])
    end
    return result
end

function objectives(this::ResultsCollector)::Vector{Number}
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




## =====================================================================================================================
## PROXIMAL GRADIENT METHOD AND THEIR VARIANTS
## =====================================================================================================================



function pgrad_map(
    f::SmoothFxn, 
    g::NonsmoothFxn, 
    eta::Number, 
    x::AbstractArray
)::AbstractArray
return x - prox_grad(f, g, eta, x) end


"""
function performs Lipschitz line search and a step of proximal gradient at the same time. 
It tests on whether (L/2)|| x - T_L(x) ||^2 >= D_f(x, T_L(x)), and L = eta^(-1), with input `x` fixed. 

## Positional Arguments

## Named Arguments


## Returns
1. `Tuple{Real, AbstractVector}`: Success and the next iterates with the new stepsize eta is returned. 
2. `Nothing`: Line search failed to produce stepsize larger than the smallest tolerance. 
3. 

"""
function prox_grad(
    f::SmoothFxn, 
    g::NonsmoothFxn,
    eta::Real, 
    x::AbstractVector,
    line_search::Bool=false,
    tol=1e-10; 
    # named arguments
    lazy::Bool=false
)::Union{Tuple{Real, AbstractVector}, Nothing}
    
    if line_search
        d(y) = f(y) - f(x) - dot(grad(f, x), y - x) # <-- Bregman divergence of the smooth part. 
        y = prox(g, eta, x - eta*grad(f, x))
        dotted = dot(y - x, y - x)
        _breg_div = d(y)
        if lazy
            # lazy then just update the Lipschitz constant, and not the future iterates. 
            eta = 0.5*dotted/_breg_div
        else
            while eta >= tol && 2*eta*_breg_div > dotted 
                y = prox(g, eta, x - eta*grad(f, x))
                dotted = dot(y - x, y - x)
                _breg_div = d(y)
                if min(dotted, _breg_div) <= eps(Float64)
                    break
                end
                eta /= 2
            end
        end
        if eta >= tol
            return eta, y
        end
        @warn "Line search failed, final η: $eta; ||x - y||^2=$dotted. D_f(x, y)=$(d(y)), div ratio: $(d(y)/dotted). "*
        "\nVerifiation criteria: `2*eta*_breg_div <= dotted evaluates` to "*
        "$(2*eta*_breg_div) <= $(dotted). "
        return nothing
    end

    return eta, prox(g, eta, x - eta*grad(f, x))
    
end


# ==============================================================================
# Full proximal gradient algorithm starts here. 
# ==============================================================================


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
- `lipschitz_line_search`: Wether to employ simple line search subroutine for 
determining the stepsize at each iteratin. 
"""
function ista(
    f::SmoothFxn, 
    g::NonsmoothFxn, 
    x0::AbstractArray;
    # named arguments 
    result_collector::ResultsCollector=ResultsCollector(),
    tol::Number=1e-18,
    max_itr::Int=2000, 
    eta::Number=1, 
    lipschitz_line_search::Bool=true
)::ResultsCollector
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
        # Proximal gradient step. 
        result = prox_grad(f, g, eta, xPre, lipschitz_line_search)
        if isnothing(result) 
            flag = 2
            break # <-- Line search fucking search failed. 
        end
        eta, x = result
        # results collection
        fxn_val, pgradMapVec = nothing, 1/eta*(x - xPre)
        if give_fxnval_nxt_itr(result_collector)
            fxn_val = f(x) + g(x)
        end
        register!(
            result_collector, 
            x, 
            eta, 
            pgradMapVec, 
            fxn_val
        )
        # Termination criteria
        if norm(x - xPre) < tol
            break   # <-- Tolerance reached, flag: 0
        end
        # Maximum iteration exceeded. 
        if k == max_itr
            flag = 1 # <-- Maxmum iteration reached. 
        end
    end
    
    result_collector.flag = flag
    return result_collector
end


"""
The VIFISTA routine is a constant acceleration, constant stepsize method for function with quadratic growth 
parameter μ and Lipschitz constant on gradient L. 
This one doesn't support line search. 

## Poisitional Arguments
- `f::SmoothFxn`: 
- `g::NonsmoothFxn`: 
- `x0::AbstractArray`: 
- `lipschitz_constant::Real`: 
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
    scnvx_const::Real;
    # Named arguments 
    result_collector::ResultsCollector=ResultsCollector(), 
    tol::Number=1e-8,
    max_itr::Int=2000
)::ResultsCollector

    @assert scnvx_const <= lipschitz_constant && scnvx_const >= 0 "Expect `sc_constant` <= `lipschitz_constant`"*
    " and `sc_constant` = 0, however this is not true and we have: "*
    "`sc_constant`=$scnvx_const, `lipschitz_constant`=$lipschitz_constant. "
    L, μ = lipschitz_constant, scnvx_const
    κ = L/μ
    η = L^(-1)
    # initiate
    x, y = x0, x0
    fxnInitialVal = collect_fxn_vals(result_collector) ? f(x) + g(y) : nothing
    initiate!(result_collector, x0, η, fxnInitialVal)
    # iterates
    flag = 0
    for k = 1:max_itr
        _, x⁺= prox_grad(f, g, η, y)
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
        if norm(x⁺ - y) < tol
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
    x0::AbstractArray; 
    lipschitz_constant::Real=1, 
    # Named arguments 
    result_collector::ResultsCollector=ResultsCollector(), 
    lipschitz_line_search::Bool=true, 
    tol::Number=1e-8,
    max_itr::Int=2000, 
    mono_restart::Bool=false
)::ResultsCollector
    L = lipschitz_constant
    N = max_itr
    ϵ = tol
    t = 1
    x, y = x0, x0
    # initiate results collector 
    _initial_fxn_val = collect_fxn_vals(result_collector) ? f(x) + g(y) : nothing
    _running_fxn_val = mono_restart ? f(x) + g(y) : nothing
    initiate!(result_collector, x0, L^(-1), _initial_fxn_val)
    flag = 0

    for k = 1: N
        # x is updated here 
        result = prox_grad(f, g, 1/L, y, lipschitz_line_search)
        if isnothing(result) 
            flag = 2
            break # <-- Line fucking search failed. 
        end
        eta, x⁺ = result
        L = eta^(-1)
        # extrapolated momentum update for y 
        t⁺ = sqrt(t^2 + 1/4) + 1/2
        θ = (t - 1)/t⁺
        # strore, check results
        _new_fxn_val, pgradMapVec = nothing, L*(x⁺ - y)
        if give_fxnval_nxt_itr(result_collector) || mono_restart
            _new_fxn_val = f(x⁺) + g(x⁺)
            if mono_restart
                _running_fxn_val = min(_new_fxn_val, _running_fxn_val)
            end
        end

        # forced restart triggered
        if mono_restart && _running_fxn_val < _new_fxn_val
            y⁺ = x + (t/t⁺)*(x - x⁺)
            x⁺ = x # reset 
        else
            y⁺ = x⁺ + θ*(x⁺ - x)
        end

        register!(
            result_collector, 
            x⁺, 
            1/L, 
            pgradMapVec, 
            _new_fxn_val
        )
        if norm(x⁺ - y) < ϵ
            break
        end
        
        # updates the iterates 
        x, y = x⁺, y⁺
        t = t⁺
        if k == max_itr
            flag = 1
            # max iteration reached
        end
    end
    
    result_collector.flag = flag
    return result_collector

end




function inexact_vfista(
    f::SmoothFxn, 
    g::NonsmoothFxn, 
    x0::AbstractArray,
    lip_const::Real=1,
    scnvx_const::Real=0;
    # Named arguments 
    result_collector::ResultsCollector=ResultsCollector(), 
    tol::Number=1e-8,
    max_itr::Int=2000, 
    estimate_scnvx_const::Bool=false, 
    lipschitz_line_search::Bool=false
)::ResultsCollector

    @assert scnvx_const <= lip_const && scnvx_const >= 0 "Expect `sc_constant` <= `lipschitz_constant`"+
    " and `sc_constant` = 0, however this is not true and we have: "+
    "`sc_constant`=$scnvx_const, `lipschitz_constant`=$lip_const. "
    L, μ = lip_const, scnvx_const
    κ = L/μ
    # initiate
    x, y = x0, x0 # current ieration of x, y
    ỹ = y # previous iteration of y 
    # t = (n) -> (n + 1)/2
    _initial_fxn = collect_fxn_vals(result_collector) ? f(x) + g(y) : nothing
    _all_scnvx_consts = Vector{Number}(); push!(_all_scnvx_consts, μ)
    initiate!(result_collector, x0, 1/L, _initial_fxn)
    _running_fval = estimate_scnvx_const ? _initial_fxn : nothing  # <--- changing in loop. 
    # iterate
    flag = 0
    for k = 1:max_itr
        # Proximal gradient on y. 
        _returned = prox_grad(f, g, 1/L, y, lipschitz_line_search)
        if isnothing(_returned)
            flag = 2 
            break  # <-- Lipschitz line search failed. 
        end
        η, x⁺ = _returned 
        L = 1/η  # update Lipschitz constant and iterates
        
        if estimate_scnvx_const
            _new_running_fval = f(x⁺)
            Df = _running_fval - _new_running_fval - dot(grad(f, x⁺), x - x⁺)
            μ⁺ = min(max(2*Df/dot(x - x⁺, x - x⁺), 0), L/2)
            μ = isnan(μ⁺) ? μ : (μ⁺ + μ)/2
            @assert !isnan(μ) # <-- Really shouldn't have Nan here. 
            _running_fval = _new_running_fval
        end
        κ = L/μ
        θ = μ > 0 ? ((sqrt(κ) - 1)/(sqrt(κ) + 1)) : 0
        y⁺ = x⁺ + θ*(x⁺ - x)
        push!(_all_scnvx_consts, μ)
        fxn_val, pgradMapVec = nothing, (1/L)*(x⁺ - y)
        if give_fxnval_nxt_itr(result_collector)
            fxn_val = (isnothing(_running_fval) ? f(x⁺) : _running_fval) + g(x⁺)
        end
        register!(
            result_collector, 
            x⁺, 
            1/L, 
            pgradMapVec, 
            fxn_val
        )
        # termination criteria
        if norm(x⁺ - y) < tol
            break # <-- Tolerance reached. 
        else
            x = x⁺
            ỹ = y
            y = y⁺
        end
        # max iter flagging. 
        if k == max_itr
            flag = 1 # <- Max itr reached. 
            break
        end
    end
    
    result_collector.flag = flag
    result_collector.misc = _all_scnvx_consts
    return result_collector
end


## =====================================================================================================================
## R-WAPG ALGORITHM NOW FOLLOWS
## =====================================================================================================================


"""
Perform R-WAPG algorithm for exactly one step and return the result vectors. 
It does the following thing: 
1. Generate x⁺, the next step. 
2. Extrapolate y⁺ using momentum rules specified by the R-WAPG algorithm. 
3. Perform Lipschitz line search if specified. 

This function should only be used by `rwapg_vfista`. 

## Positional Arguments
- `f::SmoothFxn`
- `g::NonsmoothFxn`
- `x::Vector{Number}`
- `y::Vector{Number}`
- `alpha1::Number`
- `rho::Number`
- `mu::Number`
- `L::Number`
## Named argument

## Returns
If Lipschitz line search went through: `x⁺, y⁺, α⁺, L`; else: `Nothing`


"""
function inner_rwapg(
    # Positional arguments. 
    f::SmoothFxn, 
    g::NonsmoothFxn,
    x::AbstractArray, 
    y::AbstractArray, 
    alpha1::Number, 
    rho::Number,
    mu::Number, 
    L::Number; 
    # Optinal arguments. 
    lipschitz_line_search::Bool=false, 
)::Union{Tuple{AbstractArray, AbstractArray, Number, Number}, Nothing}
    #check and assert conditions. 
    @assert alpha1 > 0 && alpha1 <= 1 "Parameter alpha1 not in range. "*
    "Condition: \"alpha1 = $alpha1 ∈ (0,1]\" FALSE. "
    @assert 0 <= (alpha1^2)*rho <= 1 "Parameter rho, alpha1 fails to adhere condition \"0 <= alpha1*rho < 1\""*
    "It has instead: alpha1=$alpha1, rho=$rho. "
    @assert 0 <= mu && mu <= L "mu, L, the strong convexity and smoothness parameters are faulty. "*
    "They violated the constaint 0 <= μ <= L. We had: μ = $mu, L = $L as the inputs. "
    r = rho*alpha1^2
    ρ = rho
    α = alpha1
    κ = mu/L
    α⁺ = (1/2)*(κ - r + sqrt((κ - r)^2 + 4r))
    θ = ρ*α*(1 - α)/(ρ*α^2 + α⁺)
    returned = prox_grad(f, g, 1/L, y, lipschitz_line_search)
    if isnothing(returned)
        return nothing
    end
    η, x⁺ = returned
    L = 1/η
    y⁺ = x⁺ + θ*(x⁺ - x)
    return x⁺, y⁺, α⁺, L
end


"""
Function implements FISTA, with restart and line search using R-WAPG. 

## Positional arguments 
- `f::SmoothFxn`: Required. 
- `g::NonsmoothFxn`: Required
- `x0::AbstractArray`: Required
- `lip_const::Real`: Required 
- `scnvx_const::Real=0`: Optional

## Named arguments

- `result_collector::ResultsCollector=ResultsCollector()`: An instance of results collector. 
- `tol::Number=1e-8`: Tolerance for termination criteria, which is the gradient mapping. 
- `max_itr::Int=2000`: Maximum number of iterations allowed. 
- `lipschitz_line_search::Bool=false`: Whether to perform line search for the Lipschitz constant. 

"""
function rwapg(
    # Basic positional arguments
    f::SmoothFxn, 
    g::NonsmoothFxn, 
    x0::AbstractArray, 
    lip_const::Real,
    scnvx_const::Real=0,
    rho::Real=1;
    # Named arguments set: Algorithm execuation
    result_collector::ResultsCollector=ResultsCollector(), 
    tol::Number=1e-8,
    max_itr::Int=2000, 
    lipschitz_line_search::Bool=false, 
    estimate_scnvx_const::Bool=false
)
    # Initial execution parameters. 
    L = lip_const           # <-- Changing in loop. 
    μ = scnvx_const         # <-- Changing in loop. 
    α = 1                   # <-- Changing in loop. 
    ρ = rho
    ϵ = tol       
    m = max_itr 
    x = x0                  # <-- Changing in loop. 
    y = x0                  # <-- Changing in loop. 
    _flag = 0
    # results collector initialization. 
    _initial_fxn = collect_fxn_vals(result_collector) ? f(x) + g(y) : nothing
    _running_fval = estimate_scnvx_const ? _initial_fxn : nothing  # <--- changing in loop. 
    _sconvx_estimated = Vector{Number}(); push!(_sconvx_estimated, μ)
    initiate!(result_collector, x0, 1/L, _initial_fxn)
    for k = 1:m
        # proximal gradient on y for x. 
        returned = inner_rwapg(
            f, g, x, y, α, ρ, μ, L, 
            lipschitz_line_search=lipschitz_line_search
        )
        if isnothing(returned) break; _flag = 2; end
        x⁺, y⁺, α⁺, L = returned
        # s-convx const estimate. 
        if estimate_scnvx_const
            _new_running_fval = f(x⁺)
            Df = _running_fval - _new_running_fval - dot(grad(f, x⁺), x - x⁺)
            μ⁺ = min(max(2*Df/dot(x - x⁺, x - x⁺), 0), L/2)
            μ = isnan(μ⁺) ? μ : (1/2)*μ⁺ + (1/2)*μ
            push!(_sconvx_estimated, μ)
            _running_fval = _new_running_fval
        end
        # results collect
        F⁺, G = nothing, (1/L)*(x⁺ - y)
        if give_fxnval_nxt_itr(result_collector)
            F⁺ = (isnothing(_running_fval) ? f(x⁺) : _running_fval) + g(x⁺)
        end
        register!(result_collector, x⁺, 1/L, G, F⁺)
        # termination criteria, updates. 
        if norm(x⁺ - y) < ϵ
            break # <-- Tolerance reached. 
        else
            x = x⁺; y = y⁺; α = α⁺
        end
        if k == m
            _flag = 1 # <-- Maximum iteration reached. 
        end
    end
    # closing and return. 
    result_collector.flag = _flag
    result_collector.misc = _sconvx_estimated
    return result_collector
end

