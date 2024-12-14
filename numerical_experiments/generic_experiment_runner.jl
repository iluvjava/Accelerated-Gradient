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
    repeat_for=50, 
)::Tuple{Vector{NTuple}, Vector{NTuple}}
    F⁺ = true_minimum
    C = length(runnables)
    # [algorithm index][test instances][iteration k]
    _objs_list = Vector{Vector{Vector}}(undef, C)
    _grad_mapping_lists = Vector{Vector{Vector}}(undef, C)
    
    @showprogress for i in 1:repeat_for
        for j in 1:C
            _objs, _gpn = perform_collect(
                () -> runnable[j](initial_guess_generator())
            )

        end
    end

    return 
end

