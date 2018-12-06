"""Classic Batch Gaussian Process Regression (no inducing points)"""
mutable struct BatchGPRegression <: FullBatchModel
    @commonfields
    @functionfields
    @kernelfields
    gnoise::DenseArray{Float64}
    """Constructor for the full batch GP Regression"""
    function BatchGPRegression(X::AbstractArray,y::AbstractArray;Autotuning::Bool=true,nEpochs::Integer = 10,
                                    kernel=0,noise::Real=1e-3,verbose::Integer=0,ArrayType::UnionAll=Array, IndPriors::Bool=true)
            this = new()
            this.ModelType = Regression
            this.Name = "Non Sparse GP Regression with Gaussian Likelihood"
            initCommon!(this,ArrayType,X,y,IndPriors,1e-16,nEpochs,verbose,Autotuning,1,Adam());
            initFunctions!(this);
            initKernel!(this,kernel);
            this.gnoise = this.AT([noise for _ in 1:this.nLatent])
            return this;
    end
end
