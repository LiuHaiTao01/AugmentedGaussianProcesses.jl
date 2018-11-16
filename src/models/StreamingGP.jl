##Implementing Thang Bui streaming ##
mutable struct StreamingGP <: OnlineGPModel
    @commonfields
    @functionfields
    @stochasticfields
    @kernelfields
    @gaussianparametersfields
    @onlinefields
    gnoise::Float64
    oldK::Symmetric{Float64,Matrix{Float64}}
    oldinvK::Symmetric{Float64,Matrix{Float64}}
    D::Symmetric{Float64,Matrix{Float64}}
    oldZ::Matrix{Float64}
    oldm::Integer
    oldΣ::Symmetric{Float64}
    oldμ::Vector{Float64}
    function StreamingGP(X::AbstractArray,y::AbstractArray;kmeansalg::KMeansAlg=CircleKMeans(),Sequential::Bool=false,AdaptiveLearningRate::Bool=false,
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
            this.oldΣ = copy(model.Σ)
            this.oldμ = copy(model.μ)
            return this;
    end
end


function updateParameters!(model::StreamingGP,iter::Integer)
    #Set as old values
    model.oldK = copy(model.Kmm)
    model.oldinvK = copy(model.invKmm)
    model.oldZ = copy(model.inducingPoints)
    model.D = inv(Symmetric(inv(model.oldΣ)-model.oldinvK))
    model.oldm = copy(model.m)
    ### DO STUFF WITH HYPERPARAMETERS HERE
    computeMatrices!(model)
    ŷ = vcat(model.y[model.MBIndices],model.D*inv(model.oldΣ)*model.oldμ);
    Kab = kernelmatrix(model.oldZ,model.inducingPoints,model.kernel)
    Kfb = vcat(model.Knm,Kab)
    Σŷ = Matrix(Diagonal(I*model.gnoise),model.nSamplesUsed+model.oldm)
    Σŷ[model.nSamplesUsed+1:end,model.nSamplesUsed+1:end] = model.D
    A = model.invKmm*model.Kfb'*inv(Σŷ)
    model.μ = A*ŷ
    model.Σ = inv(model.invKmm+A*model.Kfb*model.invKmm)
end
