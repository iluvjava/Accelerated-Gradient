include("../src/abstract_types.jl")
include("../src/non_smooth_fxns.jl")
include("../src/smooth_fxn.jl")
include("../src/proximal_gradient.jl")
using Test, LinearAlgebra, Plots, SparseArrays, ProgressMeter, Statistics
gr() 



"""
Perform the runner and collect the result out of it. 
The runner is a parameterless callable function that returns `::ResultsCollector`. 
"""
function perform_collect(
    runnable::Base.Callable
)::Tuple{Vector{Number}, Vector{Number}}
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
    initial_guess_generator::Function, 
    runnables::Vector{Function};
    true_minimum::Union{Real, Nothing}=nothing,
    repeat=50, 
    normalize::Bool=true
)::Dict
    C = length(runnables)
    _to_return = Dict{Int, Tuple}()  # <-- One parameter to handle the return of values. 
    # [algorithm index][test instances][iteration k]
    _objs_list = Vector{Vector{Vector}}(undef, C)
    _grad_mapping_lists = Vector{Vector{Vector}}(undef, C)
    _estimated_min = Inf
    for c in 1: C # for each algorithm. 
        _algo = runnables[c]
        _l1 = _objs_list[c] = Vector{Vector}()
        _l2 = _grad_mapping_lists[c] = Vector{Vector}()
        @showprogress for j in 1:repeat # repeat with different initial guesses. 
            _objs, _gpm = perform_collect(
                () -> (
                    _algo(initial_guess_generator())
                ) # <-- Pass functional handler. 
            )
            push!(_l1, _objs)
            _estimated_min = min(minimum(_objs), _estimated_min)
            push!(_l2, _gpm)
        end
    end
    if isnothing(true_minimum)
        true_minimum = _estimated_min 
    end
    true_minimum = min(true_minimum, _estimated_min) - eps(Float64)
    for c in 1: C
        if normalize
            _objs_list[c] = [
                (item .- true_minimum)/(item[1] - true_minimum) 
                for 
                item in _objs_list[c]
            ]
        end
        _to_return[c] = (
            _objs_list[c]|>zagged_arr_quantiles,
            _grad_mapping_lists[c]|>zagged_arr_quantiles
        )
    end

    return _to_return
end




"""
Given an array of array of different length: `Array{Array}`.
Get the 5 points statistics for all the kth element of all the arrays in the array. 

"""
function zagged_arr_quantiles(
    zagged_arr::Vector{Vector}
)::Vector{NTuple{5, Number}}
    _inner_max_length = zagged_arr .|> length|>maximum
    _qstats_list = Vector{NTuple{5, Number}}()
    _q5 = [0, 0.25, 0.5, 0.75, 1]
    for k in 1:_inner_max_length
        _collected = [item[k] for item in zagged_arr if k <= length(item)]
        push!(_qstats_list, tuple(quantile(_collected, _q5)...))
    end
    return _qstats_list
end
