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
    Da::Symmetric{Float64,Matrix{Float64}}
    oldZ::Matrix{Float64}
    oldm::Integer
    oldΣ::Symmetric{Float64}
    oldμ::Vector{Float64}
    L
    invL
    invD
    Σŷ
    invΣŷ
    Kfb
    ŷ
    A
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
            this.oldΣ = copy(this.Σ)
            this.oldμ = copy(this.μ)
            return this;
    end
end


function updateParameters!(model::StreamingGP,iter::Integer)
    #Set as old values
    println(iter," ",model.m)
    model.oldK = copy(model.Kmm)
    model.oldinvK = copy(model.invKmm)
    model.oldZ = copy(model.kmeansalg.centers)
    model.oldΣ= copy(model.Σ)
    model.oldm = copy(model.m)
    model.oldμ = copy(model.μ)
    model.Da = Symmetric(inv(model.oldΣ)-model.oldinvK)
    update_points!(model)
    ### DO STUFF WITH HYPERPARAMETERS HERE
    computeMatrices!(model)
    model.L = cholesky(model.Kmm)
    model.invL = inv(model.L)
    model.ŷ = vcat(model.y[model.MBIndices],model.Da\(model.oldΣ\model.oldμ));
    Kab = kernelmatrix(model.oldZ,model.kmeansalg.centers,model.kernel)
    Kab[1:model.oldm,1:model.oldm] = Kab[1:model.oldm,1:model.oldm] + jittering*I
    # if model.m >= 2 && model.oldm >= 2
    # display(heatmap(Kab))
    # sleep(0.1)
    # end
    model.Kfb = vcat(model.Knm,Kab)
    model.Σŷ = Matrix(Diagonal(I*model.gnoise,model.batchsize+model.oldm))
    model.Σŷ[model.batchsize+1:end,model.batchsize+1:end] = inv(model.Da)
    model.invΣŷ = inv(model.Σŷ)
    model.A = model.invKmm*model.Kfb'/model.Σŷ
    model.invD = inv(I + inv(model.L)*model.Kfb'*model.invΣŷ*model.Kfb*inv(model.L)+jittering*I)
    invΣ = Symmetric(model.invKmm+model.A*model.Kfb*model.invKmm)
    model.Σ = inv(invΣ)
    model.μ = invΣ\model.A*model.ŷ
end


function predict(model::StreamingGP,X_test)
    Ksb = kernelmatrix(X_test,model.kmeansalg.centers,model.kernel)
    Kss = kerneldiagmatrix(X_test,model.kernel)
    return ms = Ksb*model.invL*model.invD*model.invL*model.Kfb*model.Σŷ*model.ŷ
end

function predictproba(model::StreamingGP,X_test)
    Ksb = kernelmatrix(X_test,model.kmeansalg.centers,model.kernel)
    Kss = kerneldiagmatrix(X_test,model.kernel)
    ms = Ksb*model.invL*model.invD*model.invL*model.Kfb*model.Σŷ
    Vss = Kss - diag(Ksb*model.invKmm*Ksb')+diag(Ksb*model.invL*model.invD*model.invL*Ksb')
end
