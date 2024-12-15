include("../src/abstract_types.jl")
include("../src/non_smooth_fxns.jl")
include("../src/smooth_fxn.jl")
include("../src/proximal_gradient.jl")
using Test, LinearAlgebra, Plots, SparseArrays, ProgressMeter, Statistics
gr() # <-- Initialize GR backend. 

## =====================================================================================================================
## FUNCTIONS FOR EXPERIMENT ROUTINES
## =====================================================================================================================

function make_quadratic_problem(
    N::Integer,
    μ::Number, 
    L::Number
)::Tuple{SmoothFxn, NonsmoothFxn}
    diagonals = vcat([0], LinRange(μ, L, N - 1))
    A = diagm(diagonals) |> sparse
    b = zeros(N)
    f = Quadratic(A, b, 0)
    g = MAbs(0.0)
    return f, g
end


"""
Perform one run of multiple algorithm of interests, collect all objective value of the functions 
for each algorithm: "V-FISTA with constant parameters", "Parameter Free R-WAPG". 

## Notes
Results collector from running the algorithm is discarded by dropping the reference 
so it's not a memory hog. So that would mean: 
1. It doesn't check termination flag and just assume the algorithm terminated by reaching the tolerance. 

## Positional arguments 

## Named arguments
"""
function perform_one_experiment_on(
    f::SmoothFxn, 
    g::NonsmoothFxn, 
    x0::AbstractArray, 
    mu::Real, 
    L::Real; 
    max_itr::Int=5000, tol=1e-8
)::NTuple{2, Vector}
    μ = mu
    _results1 = vfista(
        f, 
        g, 
        x0, 
        L, 
        μ, 
        tol=tol, 
        max_itr=max_itr
    )
    _results2 = rwapg(
        f, 
        g, 
        x0, 
        L, 
        L/2, 
        lipschitz_line_search=true, 
        estimate_scnvx_const=true,
        tol=tol, 
        max_itr=max_itr
    )
    _fxnVal1 = objectives(_results1)
    _fxnVal2 = objectives(_results2)
    return _fxnVal1, _fxnVal2
end



"""
Repeat `perform_one_experiment_on` for different initial gusses and collect statistics on objective values 
of the function at each iteration of the algorithm and return the statistics information. 

## Positional Argument 
- `f::SmoothFxn`, 
- `g::NonsmoothFxn`, 
- `mu::Real`, 
- `L::Real`, 
- `true_minimum::Real`, 
- `repeat_for=100`,
- `initial_guess_generator::Base.Callable`; 
"""
function repeat_experiments_for(
    # positional argument:
    f::SmoothFxn, 
    g::NonsmoothFxn, 
    mu::Real, 
    L::Real, 
    true_minimum::Real, 
    initial_guess_generator::Base.Callable, 
    repeat_for=50, 
    # named argument: 
    max_itr::Int=5000
)::Tuple{Vector{NTuple}, Vector{NTuple}}
    μ = mu
    F⁺ = true_minimum
    _opt_gaps1, _opt_gaps2 = Vector{Vector}(), Vector{Vector}()
    @showprogress for i = 1:repeat_for
        _objs1, _objs2 = perform_one_experiment_on(
            f, g, 
            initial_guess_generator(), μ, L, max_itr=max_itr
        )
        # push in the relative error from the first initial guess
        push!(_opt_gaps1, (_objs1 .- F⁺)/_objs1[1]) 
        push!(_opt_gaps2, (_objs2 .- F⁺)/_objs2[1])
    end
    _qstats = [0, 0.25, 0.5, 0.75, 1]  # 5 point stats 
    _qstats_per_itr_1 = Vector{NTuple{_qstats|>length, Number}}()
    for k in 1: maximum(_opt_gaps1 .|> length)
        _entries = [_opt_gaps1[i][k] for i in 1: (_opt_gaps1|>length) if k <= length(_opt_gaps1[i])]
        push!(_qstats_per_itr_1, tuple(quantile(_entries, _qstats)...))
    end
    # literally horrid code, copied twice. 
    _qstats_per_itr_2 = Vector{NTuple{_qstats|>length, Number}}()
    for k in 1:maximum(_opt_gaps2 .|> length)
        _entries = [_opt_gaps2[i][k] for i in 1: (_opt_gaps2|>length) if k <= length(_opt_gaps2[i])]
        push!(_qstats_per_itr_2, tuple(quantile(_entries, _qstats)...))
    end
    return _qstats_per_itr_1, _qstats_per_itr_2
end


# Perform experiments in the following soft scope. 
begin
    N, μ, L = 512, 1e-5, 1
    f, g = make_quadratic_problem(N, μ, L)
    GenerateInitialGuess = () -> randn(N)
    Qstats1, Qstats2 = repeat_experiments_for(f, g, μ, L, 0, GenerateInitialGuess)
end

# Plots, with the Ribbons. 
begin
  
    LogMedians = [item[3] for item in Qstats1] .|> log2
    LogLowBnd = [item[1] for item in Qstats1] .|> log2
    LogUpper = [item[5] for item in Qstats1] .|> log2
    fig1 = plot(
        LogMedians; 
        ribbon = (LogMedians - LogLowBnd, LogUpper - LogMedians),
        label="V-FISTA",
        title="Simple regression (N=$N)", 
        # yaxis=:log2, 
        size=(600, 400), 
        linewidth=2, 
        ylabel="(F(x_k) - F*)/(F(x_0) - F*)", 
        xlabel="Iteration Counter", 
        dpi=300
    )
    
    LogMedians = [item[3] for item in Qstats2] .|> log2
    LogLowBnd = [item[1] for item in Qstats2] .|> log2
    LogUpper = [item[5] for item in Qstats2] .|> log2
    plot!(
        fig1, 
        LogMedians; 
        ribbon = (LogMedians - LogLowBnd, LogUpper - LogMedians),
        label="Free R-WAPG", 
        linewidth=2
    )
    fig1 |> display
end