using AugmentedGaussianProcesses.KernelModule
using Profile, ProfileView, BenchmarkTools, Test
using Random: seed!
using MLKernels, Statistics, LinearAlgebra
using ForwardDiff
seed!(42)
dims= 2
A = rand(1000,dims)
B = rand(100,dims)
#Compare kernel results with MLKernels package

#RBF Kernel
θ = 0.5
mlk = MLKernels.SquaredExponentialKernel(0.5/θ^2)
agpk = KernelModule.SEKernel(θ)
mlK = MLKernels.kernelmatrix(mlk,A)
agpK = KernelModule.kernelmatrix(A,agpk)
mlKab = MLKernels.kernelmatrix(mlk,A,B)
agpKab = KernelModule.kernelmatrix(A,B,agpk)
@testset "RBFKernel Tests" begin
    @test sum(abs.(mlK-agpK)) ≈ 0 atol = 1e-5
    @test sum(abs.(mlKab-agpKab)) ≈ 0 atol = 1e-5
end
#Matern3_2Kernel

θ = 0.5
mlk = MLKernels.MaternKernel(2.0,θ)
agpk = KernelModule.MaternKernel(θ,2.0)
mlK = MLKernels.kernelmatrix(mlk,A)
agpK = KernelModule.kernelmatrix(A,agpk)
mlKab = MLKernels.kernelmatrix(mlk,A,B)
agpKab = KernelModule.kernelmatrix(A,B,agpk)

@testset "MaternKernel Tests" begin
    @test sum(abs.(mlK-agpK)) ≈ 0 atol = 1e-5
    @test sum(abs.(mlKab-agpKab)) ≈ 0 atol = 1e-5
end

#Check for derivatives
θ = 1.0; ϵ=1e-7; ν=2.0; Aeps = copy(A); Aeps[1] = Aeps[1]+ϵ; atol=1e-5
for params in [([θ,θ],[θ+ϵ,θ]),(θ,θ+ϵ)]
    # println("Testing params $params")
    for model in [KernelModule.MaternKernel,RBFKernel]
        # println("Testing model $model")
        if model == KernelModule.MaternKernel
            k = model(params[1],ν)
            keps = model(params[2],ν)
        elseif model == KernelModule.RBFKernel
            k = model(params[1])
            keps = model(params[2])
        end
        Kmm = Symmetric(KernelModule.kernelmatrix(A,k))
        Knm = KernelModule.kernelmatrix(B,A,k)
        global diffKmm  = (KernelModule.kernelmatrix(A,keps) - KernelModule.kernelmatrix(A,k))./ϵ
        global derivKmm = typeof(params[1]) <:AbstractArray ? KernelModule.kernelderivativematrix(A,k)[1] : KernelModule.kernelderivativematrix(A,k)
        global derivKmmK = typeof(params[1]) <:AbstractArray ? KernelModule.kernelderivativematrix_K(A,Kmm,k)[1] : KernelModule.kernelderivativematrix_K(A,Kmm,k)
        global diffKnm =  (KernelModule.kernelmatrix(B,A,keps) - KernelModule.kernelmatrix(B,A,k))./ϵ
        global derivKnm = typeof(params[1]) <: AbstractArray ?  KernelModule.kernelderivativematrix(B,A,k)[1] : KernelModule.kernelderivativematrix(B,A,k)
        global derivKnmK = typeof(params[1]) <:AbstractArray ? KernelModule.kernelderivativematrix_K(B,A,Knm,k)[1] : KernelModule.kernelderivativematrix_K(B,A,Knm,k)
        global diffIndmm = ((KernelModule.kernelmatrix(Aeps,k) - KernelModule.kernelmatrix(A,k))./ϵ)[1,:]
        global derivIndmm = KernelModule.computeIndPointsJmm(k,A,1,Kmm)[:,1]
        global diffIndnm = ((KernelModule.kernelmatrix(B,Aeps,k) - KernelModule.kernelmatrix(B,A,k))./ϵ)[:,1]
        global derivIndnm = KernelModule.computeIndPointsJnm(k,B,A[1,:],1,Knm)[:,1]
        testsetname = String(Symbol(model))*(typeof(params[1]) <: AbstractArray ? " ARD" : " Iso")
        @testset "$testsetname" begin
            @test all(isapprox.(derivKmm,derivKmmK))
            @test all(isapprox.(derivKnm,derivKnmK))
            @test all(isapprox.(diffKmm,derivKmm,atol=atol))
            @test all(isapprox.(diffKnm,derivKnm,atol=atol))
            @test all(isapprox.(diffIndmm,derivIndmm,atol=atol))
            @test all(isapprox.(diffIndnm,derivIndnm,atol=atol))
        end
    end
end
