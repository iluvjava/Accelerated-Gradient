include("abstract_types.jl")
include("non_smooth_fxns.jl")
include("smooth_fxn.jl")
include("proximal_gradient.jl")


using Test, LinearAlgebra, UnicodePlots


@testset "Test Set 1: Tranditional Momentum Algorithm" begin 

    """
        Test the ISTA routine a very basic problem, all settings on the results collected turned on.     
    """
    function ISTATest1()
        
        @info "Testing: ISTA TEST 1. "
        n = 100
        L, μ = 1, 1e-2
        A = Diagonal(LinRange(μ, L, n))
        b = zeros(n)
        f = Quadratic(A, b, 0)
        g = MAbs(0.01)
        x0 = 100*ones(n)
        results = ista(f, g, x0, eps=1e-2, eta=1/L, lipschitz_line_search=true)
        report_results(results)
        
        if results.flag != 0
            global results
            return false
        end
        xs = objectives(results)
        ks = 1:length(xs)
        println(lineplot(ks, xs, title="ISTA TEST1", yscale=:log2))
        
        return true # runned without obvious issue. 
    end

    function VFISTATest1()
        @info "Testing: V-FISTA TEST 1"
        n = 100
        L, μ = 1, 1e-2
        A = Diagonal(LinRange(μ, L, n))
        b = zeros(n)
        f = Quadratic(A, b, 0)
        g = MAbs(0.01)
        x0 = 100*ones(n)
        results = vfista(f, g, x0, L, μ, tol=1e-2)
        
        if results.flag != 0
            println("termination flag: $(results.flag)")
            return false
        end
        report_results(results)
        xs = objectives(results)
        ks = 1:length(xs)
        println(lineplot(ks, xs, title="VFISTA Test1",yscale=:log2))

        return true # runned without obvious issue. 
    end

    function FISTATest1()
        @info "Testing: FISTA TEST 1"
        n = 100
        L, μ = 1, 1e-3
        A = Diagonal(LinRange(μ, L, n))
        b = zeros(n)
        f = Quadratic(A, b, 0)
        g = MAbs(0.01)
        x0 = 100*ones(n)
        
        results = fista(f, g, x0, tol=1e-2)
        report_results(results)
        xs = objectives(results)
        ks = 1:length(xs)
        println(lineplot(ks, xs, title="FISTA Test1", yscale=:log2))
        if results.flag != 0
            println("termination flag: $(results.flag)")
            return false
        end

        return true # runned without obvious issue. 
    end

    function MFISTATest1()
        @info "Testing: M-FISTA TEST 1"
        n = 100
        L, μ = 1, 1e-2
        A = Diagonal(LinRange(μ, L, n))
        b = zeros(n)
        f = Quadratic(A, b, 0)
        g = MAbs(0.01)
        x0 = 100*ones(n)
        results = fista(f, g, x0, tol=1e-10, mono_restart=true)
        report_results(results)
        xs = objectives(results)
        ks = 1:length(xs)
        println(lineplot(ks, xs, title="M-FISTA TEST 1"))
        if results.flag != 0
            println("termination flag: $(results.flag)")
            return false
        end

        return true 
    end

    function InexactVFISTATest1()
        @info "Testing: InexactVFISTA TEST 1"
        n = 100
        L, μ = 1, 1e-3
        A = Diagonal(LinRange(μ, L, n))
        b = zeros(n)
        f = Quadratic(A, b, 0)
        g = MAbs(0.01)
        x0 = 100*ones(n)
        global results
        results = inexact_vfista(
            f, 
            g, 
            x0, 
            lipschitz_line_search=true, 
            estimate_scnvx_const=true, 
            tol=1e-5, 
        )
        report_results(results)
        xs = objectives(results)
        ks = 1:length(xs)
        println(lineplot(ks, xs, title="Inexact V-FISTA Test1"))
        if results.flag != 0
            println("termination flag: $(results.flag)")
            return false
        end

        return true
    end

    # sanity test 
    @test true
    @test ISTATest1()
    @test VFISTATest1()
    @test FISTATest1()
    @test MFISTATest1()
    @test InexactVFISTATest1()
end


@testset "TEST SUIT: R-WAPG" begin 
    @info "TEST SUIT: R-WAPG soft scope loaded. "

    function SanityCheck()
        return true
    end

    @test SanityCheck()
    @info "TEST SUIT: R-WAPG sanity check passed, real tests start. "

    function RWAPG()
        n = 100
        L, μ = 1, 1e-2
        A = Diagonal(LinRange(μ, L, n))
        b = zeros(n)
        f = Quadratic(A, b, 0)
        g = MAbs(0.01)
        x0 = 100*ones(n)
        results = rwapg(
            f, g, x0, 1; tol=1e-2, 
            lipschitz_line_search=true, 
            estimate_scnvx_const=true
        )
        report_results(results)
        if results.flag != 0
            global results
            return false
        end
        xs = objectives(results)
        ks = 1:length(xs)
        println(lineplot(ks, xs, title="R-WAPG"))
        return true
    end

    @test RWAPG()

end
