
#Online Gaussian Process Regression

mutable struct OnlineGPRegression <: OnlineGPModel
    @commonfields
    @functionfields
    @stochasticfields
    @kernelfields
    @gaussianparametersfields
    @onlinefields
    gnoise::Float64
    function OnlineGPRegression(X::AbstractArray,y::AbstractArray;kmeansalg::KMeansAlg=StreamOnline(),Sequential::Bool=false,AdaptiveLearningRate::Bool=false,
                                    Autotuning::Bool=false,optimizer::Optimizer=Adam(α=0.1),OptimizeIndPoints::Bool=false,
                                    nEpochs::Integer = 1000,batchsize::Integer=-1,κ_s::Float64=0.5,τ_s::Integer=0,
                                    kernel=0,noise::Real=1e-3,m::Integer=0,AutotuningFrequency::Integer=2,
                                    ϵ::Real=1e-5,μ_init::Array{Float64,1}=[0.0],SmoothingWindow::Integer=5,
                                    verbose::Integer=0)
            this = new();
            this.ModelType = Regression;
            this.Name = "Online Sparse Gaussian Process Regression";
            initCommon!(this,X,y,noise,ϵ,nEpochs,verbose,Autotuning,AutotuningFrequency,optimizer);
            this.nSamplesUsed = batchsize
            initKernel!(this,kernel);
            initFunctions!(this);
            initOnline!(this,kmeansalg,Sequential,m)
            initStochastic!(this,AdaptiveLearningRate,batchsize,κ_s,τ_s,SmoothingWindow);
            initGaussian!(this,μ_init);
            this.gnoise = noise
            return this;
    end
end
