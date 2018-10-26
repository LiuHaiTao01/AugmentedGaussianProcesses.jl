"Printing function"
function Base.show(io::IO,model::GPModel)
    print("$(model.Name) model")
end



"Update the global variational parameters of the linear models"
function global_update!(model::LinearModel,grad_1::Vector,grad_2::Matrix)
    model.η_1 = (1.0-model.ρ_s)*model.η_1 + model.ρ_s*grad_1;
    model.η_2 = (1.0-model.ρ_s)*model.η_2 + model.ρ_s*grad_2 #Update of the natural parameters with noisy/full natural gradient
    model.μ = -0.5*model.η_2\model.η_1 #Back to the normal distribution parameters (needed for α updates)
    model.Σ = -0.5*inv(model.η_2);
end

"Update the global variational parameters of the linear models"
function global_update!(model::FullBatchModel)
    model.Σ = -0.5*inv(model.η_2);
    model.μ = model.Σ*model.η_1 #Back to the normal distribution parameters (needed for α updates)
end

"Update the global variational parameters of the sparse GP models"
function global_update!(model::SparseModel,grad_1::Vector,grad_2::Matrix)
    model.η_1 = (1.0-model.ρ_s)*model.η_1 + model.ρ_s*grad_1;
    model.η_2 = (1.0-model.ρ_s)*model.η_2 + model.ρ_s*grad_2 #Update of the natural parameters with noisy/full natural gradient
    model.Σ = -inv(model.η_2)*0.5;
    model.μ = model.Σ*model.η_1 #Back to the normal distribution parameters (needed for α updates)
end

function GaussianKL(model::FullBatchModel)
    return 0.5*(sum(model.invK.*(model.Σ+model.μ*transpose(model.μ)))-model.nSamples-logdet(model.Σ)-logdet(model.invK))
end

function GaussianKL(model::SparseModel)
    return 0.5*(sum(model.invKmm.*(model.Σ+model.μ*transpose(model.μ)))-model.m-logdet(model.Σ)-logdet(model.invKmm))
end

"Return a function computing the gradient of the ELBO given the kernel hyperparameters for full batch Models"
function hyperparameter_gradient_function(model::FullBatchModel)
    A = model.invK*(model.Σ+model.µ*transpose(model.μ))-Diagonal{Float64}(I,model.nSamples)
    #return gradient functions for lengthscale, variance and noise
    return (function(J)
                V = model.invK*J
                return 0.5*sum(V.*transpose(A))
            end,
            function(kernel)
                return 0.5/getvariance(kernel)*tr(A)
            end,
            function()
                return 0.5*sum(model.invK.*transpose(A))
            end
            )
end
