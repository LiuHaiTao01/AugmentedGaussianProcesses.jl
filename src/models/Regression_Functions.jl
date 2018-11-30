#Specific functions of the Gaussian Process regression models

"""Local updates for regression (empty)"""
function local_update!(model::BatchGPRegression)
end

"""Local updates for regression (empty)"""
function local_update!(model::SparseGPRegression)
    model.gnoise .=  broadcast((y,κ,μ,Σ)->
        1.0/model.nSamplesUsed*(
        dot(y[model.MBIndices],y[model.MBIndices])
        -2.0*dot(y[model.MBIndices],κ*μ)
        +sum((κ'*κ).*(μ*μ'+Σ))),
        model.y,model.κ,model.μ,model.Σ)
end

"Update the variational parameters of the full batch model for GP regression (empty)"
function variational_updates!(model::BatchGPRegression,iter::Integer)
    #Nothing to do here
end

"""Natural gradient computation for the sparse case"""
function natural_gradient(model::SparseGPRegression)
    grad_1 = broadcast((κ,y,σ)->model.StochCoeff.*(κ'*y[model.MBIndices])./σ,
                        model.κ,model.y,model.gnoise)
    grad_2 = broadcast((κ,σ,invK)->Symmetric(-0.5*(model.StochCoeff*κ'*κ./σ+invK)),
                        model.κ,model.gnoise,model.invK)
    return (grad_1,grad_2)
end


"""ELBO function for the basic GP Regression"""
function ELBO(model::BatchGPRegression)
    return -ExpecLogLikelihood(model)
end

"""ELBO function for the sparse variational GP Regression"""
function ELBO(model::SparseGPRegression)
    ELBO_v = model.StochCoeff*ExpecLogLikelihood(model)
    ELBO_v -= GaussianKL(model)
    return -ELBO_v
end

"""Return the expectation of the loglikelihood"""
function ExpecLogLikelihood(model::BatchGPRegression)
    return -0.5*dot(model.y,model.invK*model.y)+0.5*logdet(model.invK)-0.5*model.nSamples*log(2π)
end

"""Return the expectation of the loglikelihood for the sparse model"""
function ExpecLogLikelihood(model::SparseGPRegression)
    return -0.5*model.nSamplesUsed*sum(broadcast((σ,y,κ,μ,K̃,Σ)->log(2π*σ)
    + (sum((y[model.MBIndices]-κ*μ).^2)
    + sum(K̃)+sum((κ*Σ).*κ))/σ,model.gnoise,model.y,model.κ,model.μ,model.Ktilde,model.Σ))
end

"""Return functions computing the gradients of the ELBO given the kernel hyperparameters for a Regression Model"""
function hyperparameter_gradient_function(model::BatchGPRegression)
    A = model.invK*(model.y*transpose(model.y))-Diagonal{Float64}(I,model.nSamples)
    return (function(Jmm)
                V = model.invK*Jmm
                return 0.5*sum(V.*transpose(A))
            end,
            function(kernel)
                return 0.5/getvariance(kernel)*tr(A)
            end,
            function()
                return 0.5*sum(model.invK.*transpose(A))
            end)
end

"Return functions computing the gradients of the ELBO given the kernel hyperparameters for a sparse regression Model"
function hyperparameter_gradient_function(model::SparseGPRegression)
    F2 = model.μ*transpose(model.μ) + model.Σ
    return (function(Jmm,Jnm,Jnn)
            ι = (Jnm-model.κ*Jmm)*model.invKmm
            Jtilde = Jnn - sum(ι.*model.Knm,dims=2) - sum(model.κ.*Jnm,dims=2)
            V = model.invKmm*Jmm
            return 0.5*(sum( (V*model.invKmm).*F2)
            - model.StochCoeff/model.gnoise*sum((ι'*model.κ + model.κ'*ι).*F2) - tr(V)
            - model.StochCoeff/model.gnoise*sum(Jtilde)
            + 2*model.StochCoeff/model.gnoise*dot(model.y[model.MBIndices],ι*model.μ))
            end,
            function(kernel)
                return 0.5/(getvariance(kernel))*(sum(model.invKmm.*F2)-model.m-model.StochCoeff/model.gnoise*sum(model.Ktilde))
            end,
            function()
                ι = -model.κ*model.invKmm
                Jtilde = ones(Float64,model.nSamplesUsed) - sum(ι.*model.Knm,dims=2)[:]
                V = model.invKmm
                return 0.5*(sum( (V*model.invKmm).*F2)
                - model.StochCoeff/model.gnoise*sum((ι'*model.κ + model.κ'*ι).*F2) - tr(V)
                - model.StochCoeff/model.gnoise*sum(Jtilde)
                + 2*model.StochCoeff/model.gnoise*dot(model.y[model.MBIndices],ι*model.μ))
            end)
end

"""Return a function computing the gradient of the ELBO given the inducing point locations"""
function inducingpoints_gradient(model::SparseGPRegression)
        gradients_inducing_points = zero(model.inducingPoints)
        F2 = model.μ*transpose(model.μ) + model.Σ
        for i in 1:model.m #Iterate over the points
            Jnm,Jmm = computeIndPointsJ(model,i)
            for j in 1:model.nDim #iterate over the dimensions
                ι = (Jnm[j,:,:]-model.κ*Jmm[j,:,:])*model.invKmm
                Jtilde = -sum(ι.*model.Knm,dims=2)-sum(model.κ.*Jnm[j,:,:],dims=2)
                V = model.invKmm*Jmm[j,:,:]
                gradients_inducing_points[i,j] = 0.5*(sum( (V*model.invKmm).*F2)
                - model.StochCoeff/model.gnoise*sum((ι'*model.κ + model.κ'*ι).*F2) - tr(V) - model.StochCoeff/model.gnoise*sum(Jtilde)
                + 2*model.StochCoeff/model.gnoise*dot(model.y[model.MBIndices],ι*model.μ))
            end
        end
        return gradients_inducing_points
end


"""Take the labels y and adapt the model to it"""
function treatlabels!(model::Union{BatchGPRegression,SparseGPRegression},y::DenseArray{T,N}) where {T,N}
    if N > 2
        @error "Cannot use a tensor of labels"
    elseif N == 2
        (n,d) = size(y)
        model.nLatent = d
        model.y = model.AT([y[:,i] for i in 1:d])
    else
        model.nLatent = 1
        model.y = model.AT([y])
    end
end
