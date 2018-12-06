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
    return -0.5*sum(broadcast((y,invK)->dot(y,invK*y)+logdet(invK)-model.nSamples*log(2π),model.y,model.invK))
end

"""Return the expectation of the loglikelihood for the sparse model"""
function ExpecLogLikelihood(model::SparseGPRegression)
    return -0.5*sum(broadcast((σ,y,κ,μ,K̃,Σ)->log(2π*σ)
    + (sum((y[model.MBIndices]-κ*μ).^2)
    + sum(K̃)+sum((κ*Σ).*κ))/σ,model.gnoise,model.y,model.κ,model.μ,model.Ktilde,model.Σ))
end

"""Return functions computing the gradients of the ELBO given the kernel hyperparameters for a Regression Model"""
function hyperparameter_gradient_function(model::BatchGPRegression)
    A = model.invK.*(model.y.*transpose(model.y)).-[Diagonal(I,model.nSamples)]
    if model.IndPriors
        return (function(Jmm,i,_)
                    V = model.invK[i]*Jmm
                    return 0.5*sum(V.*transpose(A[i]))
                end,
                function(kernel,i,_)
                    return 0.5/getvariance(kernel)*tr(A[i])
                end)
    else
        return (function(Jmm,_,_)
                    V = model.invK[1]*Jmm
                    return 0.5*sum([sum(V.*transpose(a))] for a in A)
                end,
                function(kernel,_,_)
                    return 0.5/getvariance(kernel)*sum(tr.(A))
                end)
    end
end

"Return functions computing the gradients of the ELBO given the kernel hyperparameters for a sparse regression Model"
function hyperparameter_gradient_function(model::SparseGPRegression)
    F2 = model.μ.*transpose(model.μ) .+ model.Σ
    if model.IndPriors
        return (function(Jmm,Jnm,Jnn,iter,_)
                ι = (Jnm-model.κ[iter]*Jmm)*model.invK[iter]
                Jtilde = Jnn - vec(sum(ι.*model.Knm[iter],dims=2)) - vec(sum(model.κ[iter].*Jnm,dims=2))
                V = model.invK[iter]*Jmm
                return 0.5*(sum( (V*model.invK[iter]).*F2[iter])
                - model.StochCoeff/model.gnoise[iter]*sum((ι'*model.κ[iter] + model.κ[iter]'*ι).*F2[iter]) - tr(V)
                - model.StochCoeff/model.gnoise[iter]*sum(Jtilde)
                + 2*model.StochCoeff/model.gnoise[iter]*dot(model.y[iter][model.MBIndices],ι*model.μ[iter]))
                end,
                function(kernel,iter)
                    return 0.5/getvariance(kernel)*sum(model.invK[iter].*F2[iter])-model.m-model.StochCoeff/model.gnoise[iter]*sum(model.Ktilde[iter])
                end)
    else
        return (function(Jmm,Jnm,Jnn,iter,_)
                ι = (Jnm-model.κ[1]*Jmm)*model.invK[1]
                Jtilde = Jnn - vec(sum(ι.*model.Knm[1],dims=2)) - vec(sum(model.κ[1].*Jnm,dims=2))
                V = model.invK[1]*Jmm
                return 0.5*sum(broadcast((f2,σ,y,μ)->sum( (V*model.invK[1]).*f2)
                - model.StochCoeff/σ*sum((ι'*model.κ[1] + model.κ[1]'*ι).*f2) - tr(V)
                - model.StochCoeff/σ*sum(Jtilde)
                + 2*model.StochCoeff/σ*dot(y[model.MBIndices],ι*μ),F2,model.gnoise,model.y,model.μ))
                end,
                function(kernel,iter)
                    return 0.5/getvariance(kernel)*sum(broadcast((f2,σ)->sum(model.invK[1].*f2)-model.m-model.StochCoeff/σ*sum(model.Ktilde[1]),F2,model.gnoise))
                end)
    end
end

"""Return a function computing the gradient of the ELBO given the inducing point locations"""
function inducingpoints_gradient(model::SparseGPRegression)
        F2 = model.μ.*transpose(model.μ) .+ model.Σ
        if model.IndPriors
            gradients_inducing_points = [zero(model.inducingPoints[1]) for _ in 1:model.nLatent]
            for iter in in 1:model.nLatent
                for i in 1:model.m #Iterate over the points
                    Jnm,Jmm = computeIndPointsJ(model,i,iter)
                    for j in 1:model.nDim #iterate over the dimensions
                        ι = (Jnm[j,:,:]-model.κ[iter]*Jmm[j,:,:])*model.invK[iter]
                        Jtilde = -sum(ι.*model.Knm[iter],dims=2)-vec(sum(model.κ[iter].*Jnm[j,:,:],dims=2))
                        V = model.invK[iter]*Jmm[j,:,:]
                        gradients_inducing_points[i,j] = 0.5*(sum( (V*model.invK[iter]).*F2)
                        - model.StochCoeff/model.gnoise*sum((ι'*model.κ[iter] + model.κ[iter]'*ι).*F2[iter]) - tr(V) - model.StochCoeff/model.gnoise*sum(Jtilde)
                        + 2*model.StochCoeff/model.gnoise*dot(model.y[iter][model.MBIndices],ι*model.μ[iter]))
                    end
                end
            end
        else
            gradients_inducing_points = zero(model.inducingPoints[1])
            for i in 1:model.m #Iterate over the points
                Jnm,Jmm = computeIndPointsJ(model,i,iter)
                for j in 1:model.nDim #iterate over the dimensions
                    ι = (Jnm[j,:,:]-model.κ[1]*Jmm[j,:,:])*model.invK[1]
                    Jtilde = -vec(sum(ι.*model.Knm[1],dims=2))-vec(sum(model.κ[1].*Jnm[j,:,:],dims=2))
                    V = model.invK[1]*Jmm[j,:,:]
                    gradients_inducing_points[i,j] = 0.5*sum(broadcast((f2,μ,y)->sum( (V*model.invK[1]).*f2)
                    - model.StochCoeff/model.gnoise*sum((ι'*model.κ[1] + model.κ[1]'*ι).*f2) - tr(V) - model.StochCoeff/model.gnoise*sum(Jtilde)
                    + 2*model.StochCoeff/model.gnoise*dot(y[model.MBIndices],ι*μ[iter]),F2,model.μ,model.y))
                end
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
