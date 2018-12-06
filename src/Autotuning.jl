

"Update all hyperparameters for the linear models"
function updateHyperParameters!(model::LinearModel,iter::Integer)
    grad_noise = 0.5*((tr(model.Σ)+norm(model.μ))/(model.noise^2.0)-model.nFeatures/model.noise);
    model.noise += GradDescent.update(model.optimizers[1],grad_noise)
    model.HyperParametersUpdated = true
end

"Update all hyperparameters for the full batch GP models"
function  updateHyperParameters!(model::FullBatchModel)
    Jnn = [kernelderivativematrix(model.X,kernel) for kernel in model.kernel] #Compute all the derivatives of the matrix Knn given the kernel parameters
    f_l,f_v = hyperparameter_gradient_function(model)
    if model.IndPriors
        grads_l = map(compute_hyperparameter_gradient,model.kernel,[f_l for _ in 1:model.nLatent],Jnn,1:model.nLatent,1:model.nLatent)
        grads_v = map(f_v,model.kernel,1:model.nLatent,1:model.nLatent)
    else
        grads_l = compute_hyperparameter_gradient(model.kernel[1],f_l,Jnn[1],1,1)
        grads_v = [f_v(model.kernel[1],1,1)]
    end
    apply_gradients_lengthscale!.(model.kernel,grads_l) #Send the derivative of the matrix to the specific gradient of the model
    apply_gradients_variance!.(model.kernel,grads_v) #Send the derivative of the matrix to the specific gradient of the model
    model.HyperParametersUpdated = true
end

"Update all hyperparameters for the full batch GP models"
function updateHyperParameters!(model::SparseModel)
    f_l,f_v = hyperparameter_gradient_function(model)
    # Jmm = [kernelderivativematrix(Xᵤ,kernel) for (Xᵤ,kernel) in zip(model.inducingPoints,model.kernel)] #Compute all the derivatives of the matrix Kmm given the kernel
    # Jnm = [kernelderivativematrix(model.X[model.MBIndices,:],Xᵤ,kernel) for (Xᵤ,kernel) in zip(model.inducingPoints,model.kernel)]#Compute all the derivative of the matrix Knm given the kernel
    # Jnn = [kernelderivativediagmatrix(model.X[model.MBIndices,:],kernel) for kernel in model.kernel] #Compute all the derivatives of the diagonal matrix Knn given the kernel
    if model.IndPriors
        matrix_derivatives =[[kernelderivativematrix(Xᵤ,kernel),
        kernelderivativematrix(model.X[model.MBIndices,:],Xᵤ,kernel),
        kernelderivativediagmatrix(model.X[model.MBIndices,:],kernel)] for (Xᵤ,kernel) in zip(model.inducingPoints,model.kernel)]
        grads_l = map(compute_hyperparameter_gradient,model.kernel,[f_l for _ in 1:model.nLatent],matrix_derivatives,1:model.nLatent,1:model.nLatent)
        grads_v = map(f_v,model.kernel,1:model.nLatent)
    else
        matrix_derivatives = [kernelderivativematrix(model.inducingPoints[1],model.kernel[1]),
                            kernelderivativematrix(model.X[model.MBIndices,:],model.inducingPoints[1],model.kernel[1]),
                            kernelderivativediagmatrix(model.X[model.MBIndices,:],model.kernel[1])]
        grads_l = [compute_hyperparameter_gradient(model.kernel[1],f_l,matrix_derivatives,1,1)]
        grads_v = [f_v(model.kernel[1],1)]
    end
    if model.OptimizeInducingPoints
        inducingpoints_gradients = inducingpoints_gradient(model) #Compute the gradient given the inducing points location
        model.inducingPoints += GradDescent.update(model.optimizer,inducingpoints_gradients) #Apply the gradients on the location
    end
    apply_gradients_lengthscale!.(model.kernel,grads_l)
    apply_gradients_variance!.(model.kernel,grads_v)
    model.HyperParametersUpdated = true
end


"Update all hyperparameters for the multiclass GP models"
function updateHyperParameters!(model::MultiClass)
    f_l,f_v = hyperparameter_gradient_function(model)
    if model.IndependentGPs
        Jnn = [kernelderivativematrix(model.X,model.kernel[i]) for i in model.KIndices]
        grads_l = map(compute_hyperparameter_gradient,model.kernel[model.KIndices],[f_l for _ in 1:model.nClassesUsed],Jnn,model.KIndices,1:model.nClassesUsed)
        grads_v = map(f_v,model.kernel[model.KIndices],model.KIndices,1:model.nClassesUsed)
        apply_gradients_lengthscale!.(model.kernel[model.KIndices],grads_l)
        apply_gradients_variance!.(model.kernel[model.KIndices],grads_v)
    else
        Jnn = kernelderivativematrix(model.X,model.kernel[1])
        grads_l = compute_hyperparameter_gradient(model.kernel[1],f_l,Jnn,1,1)
        grads_v = f_v(model.kernel[1])
        apply_gradients_lengthscale!(model.kernel[1],grads_l)
        apply_gradients_variance!(model.kernel[1],grads_v)
    end
    model.HyperParametersUpdated = true
end


function updateHyperParameters!(model::SparseMultiClass)
    f_l,f_v = hyperparameter_gradient_function(model)
    if model.IndependentGPs
        matrix_derivatives =[[kernelderivativematrix(model.inducingPoints[kiter],model.kernel[kiter]),
        kernelderivativematrix(model.X[model.MBIndices,:],model.inducingPoints[kiter],model.kernel[kiter]),
        kernelderivativediagmatrix(model.X[model.MBIndices,:],model.kernel[kiter])] for kiter in model.KIndices]
        grads_l = map(compute_hyperparameter_gradient,model.kernel[model.KIndices],[f_l for _ in 1:model.nClassesUsed],matrix_derivatives,model.KIndices,1:model.nClassesUsed)
        # println([getvariance(k) for k in model.kernel])
        grads_v = map(f_v,model.kernel[model.KIndices],model.KIndices,1:model.nClassesUsed)
        # println("Variances grad :", grads_v)
        apply_gradients_lengthscale!.(model.kernel[model.KIndices],grads_l)
        apply_gradients_variance!.(model.kernel[model.KIndices],grads_v)
        # setvariance(model)
    else
        matrix_derivatives = [kernelderivativematrix(model.inducingPoints[1],model.kernel[1]),
                            kernelderivativematrix(model.X[model.MBIndices,:],model.inducingPoints[1],model.kernel[1]),
                            kernelderivativediagmatrix(model.X[model.MBIndices,:],model.kernel[1])]
        grads_l = compute_hyperparameter_gradient(model.kernel[1],f_l,matrix_derivatives,1,1)
        grad_v = f_v(model.kernel[1])
        apply_gradients_lengthscale!(model.kernel[1],grads_l)
        apply_gradients_variance!(model.kernel[1],grad_v)
    end
    if model.OptimizeInducingPoints
        inducingpoints_gradients = inducingpoints_gradient(model)
        model.inducingPoints += GradDescent.update(model.optimizer,inducingpoints_gradients)
    end
    model.HyperParametersUpdated = true
end


function apply_gradients_noise!(model::GPModel,grads_n)
    KernelModule.update!(model.noise,grads_n)
end
