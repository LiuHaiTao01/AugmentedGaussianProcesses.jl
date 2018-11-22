# Functions related to the Student T likelihood (HGP)
"""Update the local variational parameters of the full batch GP HGP"""
function local_update!(model::BatchHGP)
    model.λ = 0.5*((model.μ-model.y).^2+diag(model.Σ))
    model.c = sqrt.(diag(model.Σ_g)+model.μ_g.^2)
    model.γ = 0.5*model.α*model.λ./cosh.(0.5*model.c).*exp.(-0.5*model.μ_g)
    model.θ = 0.5.*(model.γ.+1.0)./model.c.*tanh.(0.5*model.c)
    model.K_g = Symmetric(kernelmatrix(model.X,model.kernel_g)+jittering*I)
    model.invK_g = inv(model.K_g)
    model.Σ_g = Symmetric(inv(Diagonal(model.θ)+model.invK_g))
    model.μ_g = 0.5*model.α*model.Σ_g*(1.0.-model.γ)
end

# """Update the local variational parameters of the sparse GP HGP"""
# function local_update!(model::SparseHGP)
#     model.β = 0.5.*(model.Ktilde+sum((model.κ*model.Σ).*model.κ,dims=2)[:]+(model.κ*model.μ.-model.y[model.MBIndices]).^2 .+model.ν)
#     model.θ = 0.5.*(model.ν.+1.0)./model.β
#
# end

"""Return the natural gradients of the ELBO given the natural parameters"""
function natural_gradient(model::BatchHGP)
    expec = expectation.(logit,Normal.(model.μ_g,diag(model.Σ_g)))
    model.η_1 = 0.5*model.α*expec.*model.y
    model.η_2 = Symmetric(-0.5*(Diagonal{Float64}(expec) + model.invK))
end

# """Return the natural gradients of the ELBO given the natural parameters"""
# function natural_gradient(model::SparseHGP)
#     grad_1 =  model.StochCoeff*model.κ'*(model.θ.*model.y[model.MBIndices])
#     grad_2 = Symmetric(-0.5.*(model.StochCoeff*transpose(model.κ)*Diagonal{Float64}(model.θ)*model.κ + model.invKmm))
#     return (grad_1,grad_2)
# end


"""Compute the negative ELBO for the full batch HGP Model"""
function ELBO(model::BatchHGP)
    ELBO_v = 0.0
    # ELBO_v = ExpecLogLikelihood(model)
    ELBO_v += -GaussianKL(model)
    # ELBO_v += -InverseGammaKL(model)
    return -ELBO_v
end

#
# """Compute the negative ELBO for the sparse HGP Model"""
# function ELBO(model::SparseHGP)
#     ELBO_v = model.StochCoeff*ExpecLogLikelihood(model)
#     ELBO_v += -GaussianKL(model)
#     ELBO_v += -model.StochCoeff*InverseGammaKL(model)
#     return -ELBO_v
# end

"""Return the expected log likelihood for the batch HGP Model"""
function ExpecLogLikelihood(model::BatchHGP)
    tot = -0.5*model.nSamples*log(2*π)
    tot -= 0.5.*(sum(log.(model.β))-model.nSamples*digamma(model.α))
    tot -= 0.5.*sum(model.α./model.β.*(diag(model.Σ)+model.μ.^2-2*model.μ.*model.y-model.y.^2))
    return tot
end

# """Return the expected log likelihood for the sparse HGP Model"""
# function ExpecLogLikelihood(model::SparseHGP)
#     tot = -0.5*model.nSamplesUsed*log(2*π)
#     tot -= 0.5.*(sum(log.(model.β))-model.nSamples*digamma(model.α))
#     tot -= 0.5.*sum(model.α./model.β.*(model.Ktilde + sum((model.κ*model.Σ).*model.κ,dims=2)[:]+(model.κ*model.μ).^2-2*(model.κ*model.μ).*model.y[model.MBIndices]-model.y[model.MBIndices].^2))
#     return tot
# end

# """Return the KL divergence for the inverse gamma distributions"""
# function InverseGammaKL(model::GPModel)
#     α_p = β_p = model.ν/2;
#     return (α_p.-model.α)*digamma(α_p).-log(gamma(α_p)).+log(gamma(model.α))
#             .+ model.α*(log(β_p).-log.(model.β)).+α_p.*(model.β.-β_p)./β_p
# end

# """Return a function computing the gradient of the ELBO given the kernel hyperparameters for a HGP Model"""
# function hyperparameter_gradient_function(model::SparseHGP)
#     F2 = Symmetric(model.μ*transpose(model.μ) + model.Σ)
#     θ = Diagonal(model.θ)
#     return (function(Jmm,Jnm,Jnn)
#                 ι = (Jnm-model.κ*Jmm)*model.invKmm
#                 Jtilde = Jnn - sum(ι.*model.Knm,dims=2)[:] - sum(model.κ.*Jnm,dims=2)[:]
#                 V = model.invKmm*Jmm
#                 return 0.5*(sum( (V*model.invKmm - model.StochCoeff*(ι'*θ*model.κ + model.κ'*θ*ι)) .* transpose(F2)) - tr(V) - model.StochCoeff*dot(model.θ,Jtilde)
#                     + model.StochCoeff*dot(model.y[model.MBIndices],ι*model.μ))
#             end,
#             function(kernel)
#                 return  0.5/(getvariance(kernel))*(sum(model.invKmm.*F2)-model.StochCoeff*dot(model.θ,model.Ktilde)-model.m)
#             end,
#             function()
#                 ι = -model.κ*model.invKmm
#                 Jtilde = ones(Float64,model.nSamplesUsed) - sum(ι.*model.Knm,dims=2)[:]
#                 V = model.invKmm
#                 return 0.5*(sum( (V*model.invKmm - model.StochCoeff*(ι'*θ*model.κ + model.κ'*θ*ι)) .* transpose(F2)) - tr(V) - model.StochCoeff*dot(model.θ,Jtilde)
#                     + model.StochCoeff*dot(model.y[model.MBIndices],ι*model.μ))
#             end)
# end

# """Return a function computing the gradient of the ELBO given the inducing point locations"""
# function inducingpoints_gradient(model::SparseHGP)
#     gradients_inducing_points = zeros(model.inducingPoints)
#     B = model.μ*transpose(model.μ) + model.Σ
#     Kmn = kernelmatrix(model.inducingPoints,model.X[model.MBIndices,:],model.kernel)
#     θ = Diagonal(0.25./model.α.*tanh.(0.5*model.α))
#     for i in 1:model.m #Iterate over the points
#         Jnm,Jmm = computeIndPointsJ(model,i)
#         for j in 1:model.nDim #Compute the gradient over the dimensions
#             ι = (Jnm[j,:,:]-model.κ*Jmm[j,:,:])/model.Kmm
#             Jtilde = -sum(ι.*(transpose(Kmn)),dims=2)[:]-sum(model.κ.*Jnm[j,:,:],dims=2)[:]
#             V = model.Kmm\Jmm[j,:,:]
#             gradients_inducing_points[i,j] = 0.5*(sum( (V/model.Kmm - model.StochCoeff*(ι'*θ*model.κ + model.κ'*θ*ι)) .* transpose(B)) - tr(V) - model.StochCoeff*dot(diag(θ),Jtilde)
#                 + model.StochCoeff*dot(model.y[model.MBIndices],ι*model.μ))
#         end
#     end
#     return gradients_inducing_points
# end
