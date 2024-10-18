include("abstract_types.jl")
include("non_smooth_fxns.jl")
include("smooth_fxn.jl")
include("proximal_gradient.jl")


using Test, LinearAlgebra, UnicodePlots

@testset "Test Set 1" begin 
    

    """
        Test the ISTA routine a very basic problem, all settings on the results collected turned on.     
    """
    function basicISTATest1()
        
        # just throw assertion error if something went wrong with the numerics of the algorithm. 
        # this function go through the ISTA algorithm with different setting as a run through of stuff
        # from most part of the code base. 
        @info "Testing: basicISTATest1"
        n = 100
        L, μ = 1, 1e-2
        A = Diagonal(LinRange(μ, L, n))
        b = zeros(n)
        f = Quadratic(A, zeros(size(b)), 0)
        g = MAbs(0.01)
        x0 = 1*ones(n)
        results = ista(f, g, x0, eps=1e-2, eta=1/L)
        if results.flag != 0
            return false
        end
        report_results(results)
        xs = get_all_objective_vals(results)
        ks = 1:length(xs)
        println(lineplot(ks, xs, title="ISTA BASIC TEST1", yscale=:log10))

        return true # runned without obvious issue. 
    end

    function basicVFISTATest1()
        @info "Testing: basicVFISTATest1"
        n = 100
        L, μ = 1, 1e-2
        A = Diagonal(LinRange(μ, L, n))
        b = zeros(n)
        f = Quadratic(A, zeros(size(b)), 0)
        g = MAbs(0.01)
        x0 = 1*ones(n)
        results = vfista(f, g, x0, L, μ, eps=1e-2)
        
        if results.flag != 0
            println("termination flag: $(results.flag)")
            return false
        end
        report_results(results)
        xs = get_all_objective_vals(results)
        ks = 1:length(xs)
        print(lineplot(ks, xs, title="VFISTA BASIC TEST1", yscale=:log10))

        return true # runned without obvious issue. 
    end

    function basicPPMAPGTest1()
        @info "Testing: basicPPMAPGTest1"
        n = 100
        L, μ = 1, 1e-2
        A = Diagonal(LinRange(μ, L, n))
        b = zeros(n)
        f = Quadratic(A, zeros(size(b)), 0)
        g = MAbs(0.01)
        x0 = 1*ones(n)
        
        global results
        results = ppm_apg(f, g, x0, eps=1e-2)
        
        report_results(results)
        xs = get_all_objective_vals(results)
        ks = 1:length(xs)
        print(lineplot(ks, xs, title="basicPPMAPGTest1 TEST1", yscale=:log10))
        
        if results.flag != 0
            println("termination flag: $(results.flag)")
            return false
        end

        return true # runned without obvious issue. 
    end


    # sanity test 
    @test true
    @test basicISTATest1()
    @test basicVFISTATest1()
    @test basicPPMAPGTest1()
end