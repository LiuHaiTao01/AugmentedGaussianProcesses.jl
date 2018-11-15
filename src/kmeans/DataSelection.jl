##Work in progress to test methods##


mutable struct DataSelection <: KMeansAlg
    accepting_function::Function ###From given parameters return if point is accepted (return true or false)
    lim
    k::Int64
    centers::Array{Float64,2}
    function DataSelection(;func=RandomAccept_Mean,lim=[0.9,0.4])
        return new(func,lim)
    end
end

function init!(alg::DataSelection,X,y,model,k::Int64)
    n = size(X,1)
    # init_k = max(1,ceil(Int64,n/10))
    init_k = 10
    alg.centers = reshape(X[sample(1:n,init_k,replace=false),:],init_k,size(X,2))
    alg.k = init_k
end

function update!(alg::DataSelection,X,y,model)
    b = size(X,1)
    new_centers = []
    new_k = 0
    for i in 1:b
        mu,sig = model.fstar(reshape(X[i,:],1,size(X,2)))
        # println("σ: $sig,\tabs(mu-f): $(1-exp(-0.5*(mu-y[i])[1]^2/sig[1])),\tquant: $(log(sqrt(sig))+0.5*(y[i]-mu[1])^2/sig[1])")
        if alg.accepting_function(alg,model,mu,sig,X[i,:],y[i])
            # μ_new,σ_new = model.predictproba(X[i,:])
            # alg.centers = vcat(alg.centers,X[i,:]')
            # update_model!(model,reshape(X[i,:],1,size(X,2)),μ_new,σ_new)
            # alg.k += 1
            new_k += 1
            push!(new_centers,X[i,:])
        end
    end
    for v in new_centers
        alg.centers = vcat(alg.centers,v)
    end
    alg.k += new_k
end

function RandomAccept_Mean(alg,model,mu,sig,X,y)
    diff_f = 1.0.-exp.(-0.5*(mu.-y)[1]^2/sig[1])
    d = find_nearest_center(X,alg.centers,model.kernel)[2]
    if d>2*(1-alg.lim[1])
        println("Distance point")
        return true
    elseif diff_f > alg.lim[2] && d<2*(1-alg.lim[1]-0.05)
        println("Likelihood point")
        return true
    end
    return false
    # println(sig[1])
    #return sig[1]>0.8*(1-diff_f)
    # return KLGP(mu[1],sig[1],y,0.001)>10
    # return (d>(1-2*alg.lim[1]) || diff_f>0.5)
    # return JSGP(mu[1],sig[1],y,0.001)>10
    # return sig[1]>0.8
    # return (0.5*sqrt(sig[1])+0.5*diff_f)>rand()
    # return 1.0*sqrt(sig[1])>rand()
end



function get_likelihood_diff(model,alg,new_point)
    kplus = kernelmatrix(alg.centers,new_point,model.kernel)
    a = model.invKmm*kplus
    c = kerneldiagmatrix(new_point,model.kernel)[1] - dot(kplus,a)
    kfu = kernelmatrix(model.X[model.MBIndices,:],new_point,model.kernel)
    b = 1.0/sqrt(c)*(kfu*model.invKmm*kplus-kfu)
    A = (1.0/model.gnoise*I-1.0/model.gnoise^2 * model.κ*model.Σ*model.κ')
    v = 1+dot(b,A*b)
    diffL = log(v)
    - 1.0/model.gnoise*tr(b*b')
    - dot(model.y[model.MBIndices],(A*b*b'*A)/v*model.y[model.MBIndices])
    return diffL
end
export LikelihoodImprovement
function LikelihoodImprovement(alg,model,mu,sig,X,y)
    println(get_likelihood_diff(model,alg,X))
    if get_likelihood_diff(model,alg,X) > alg.lim
        return true
    end
    return false
end
