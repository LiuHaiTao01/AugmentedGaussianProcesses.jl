"""Sparse Gaussian Process Regression with Gaussian Likelihood"""
mutable struct SparseGPRegression <: SparseModel
    @commonfields
    @functionfields
    @stochasticfields
    @kernelfields
    @sparsefields
    @gaussianparametersfields
    gnoise::DenseVector{Float64}
    """Constructor Sparse Gaussian Process Regression with Gaussian Likelihood"""
    function SparseGPRegression(X::AbstractArray,y::AbstractArray;Stochastic::Bool=false,AdaptiveLearningRate::Bool=true,
                                    Autotuning::Bool=false,optimizer::Optimizer=Adam(α=0.1),OptimizeIndPoints::Bool=false,
                                    nEpochs::Integer = 10000,batchsize::Integer=-1,κ_s::Real=1.0,τ_s::Integer=100,
                                    kernel=0,noise::Real=1e-3,m::Integer=0,AutotuningFrequency::Integer=2,
                                    ϵ::Real=1e-5,μ_init::Array{Float64,1}=[0.0],SmoothingWindow::Integer=5,
                                    verbose::Integer=0,ArrayType::UnionAll=Array, IndPriors::Bool=true)
            this = new();
            this.ModelType = Regression;
            this.Name = "Sparse Gaussian Process Regression with Gaussian Likelihood";
            initCommon!(this,ArrayType,X,y,IndPriors,ϵ,nEpochs,verbose,Autotuning,AutotuningFrequency,optimizer);
            initFunctions!(this);
            if Stochastic
                initStochastic!(this,AdaptiveLearningRate,batchsize,κ_s,τ_s,SmoothingWindow);
            else
                this.MBIndices = 1:this.nSamples; this.nSamplesUsed = this.nSamples; this.StochCoeff=1.0;
            end
            initKernel!(this,kernel);
            initSparse!(this,m,OptimizeIndPoints);
            initGaussian!(this,μ_init);
            if this.Stochastic && this.AdaptiveLearningRate
                MCInit!(this)
            end
            this.gnoise = this.AT([noise for _ in 1:this.nLatent])
            return this;
    end
    "Empty constructor for loading models"
    function SparseGPRegression()
        this = new()
        this.ModelType = Regression
        this.Name = "Sparse Gaussian Process Regression with Gaussian Likelihood";
        initFunctions!(this)
        return this;
    end
end
