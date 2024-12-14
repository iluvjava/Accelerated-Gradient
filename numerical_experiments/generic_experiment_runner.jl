include("../src/abstract_types.jl")
include("../src/non_smooth_fxns.jl")
include("../src/smooth_fxn.jl")
include("../src/proximal_gradient.jl")
using Test, LinearAlgebra, Plots, SparseArrays, ProgressMeter, Statistics
gr() # <-- Initialize GR backend. 


"""
Perform the runner and collect the result out of it. 
The runner is a parameterless callable function that returns `::ResultsCollector`. 
"""
function perform_collect(
    runnable::Base.Callable
)::Vector{Number}
    results::ResultsCollector = runnable()
    return results |> objectives, results.gradient_mapping_norm
end



"""
Repeat `perform_one_experiment_on` for different initial gusses and collect statistics on objective values 
of the function at each iteration of the algorithm and return the statistics information. 

## Positional Argument 
- `initial_guess_generator::Base.Callable`: 
- `runnables::Vector{Base.Callable}`:
- `true_minimum::Union{Real, Nothing}=nothing`: 
- `repeat_for=50`: 
"""
function repeat_experiments_for(  
    initial_guess_generator::Base.Callable, 
    runnables::Vector{Base.Callable},
    true_minimum::Union{Real, Nothing}=nothing;
    repeat=50, 
    normalize::Bool=false
)::Dict
    C = length(runnables)
    _to_return = Dict{Int, Tuple}()  # <-- One parameter to handle the return of values. 
    # [algorithm index][test instances][iteration k]
    _objs_list = Vector{Vector{Vector}}(undef, C)
    _grad_mapping_lists = Vector{Vector{Vector}}(undef, C)
    _estimated_min = Inf
    for c in 1: C # for each algorithm. 
        _algo = runnables(c)
        _l1 = _objs_list[c] = Vector{Vector}()
        _l2 = _grad_mapping_lists[c] = Vector{Vector}()
        for j in 1:repeat # repeat with different initial guesses. 
            _objs, _gpm = perform_collect(
                () -> (
                    _algo(initial_guess_generator())
                ) # <-- Pass functional handler. 
            )
            push!(_l1, _objs)
            _estimated_min = min(mininum(objs), _estimated_min)
            push!(_l2, _gpm)
        end
    end
    if isnothing(true_minimum)
        true_minimum = _estimated_min 
    end
    true_minimum = min(true_minimum, _estimated_min) - eps(Float64)
    if normalize
        _objs_list = [(objs .- true_minimum)/(objs[1] .- true_minimum) in _objs_list]
    end
    for c in 1: C
        _to_return[c] = (_objs_list[c], _grad_mapping_lists[c])
    end

    return _to_return
end


### EXAMPLE USAGE OF THE ABOVE GENERIC INTERFACE. 