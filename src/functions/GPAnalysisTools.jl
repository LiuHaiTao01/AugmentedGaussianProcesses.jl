module GPAnalysisTools
using ValueHistories
using LinearAlgebra
using Statistics
using Plots
export getLog, getMultiClassLog
export IntermediatePlotting
"""
    Return a a MVHistory object and callback function for the training of binary classification problems
    The callback will store the ELBO and the variational parameters at every iterations included in iter_points
    If X_test and y_test are provided it will also store the test accuracy and the mean and median test loglikelihood
"""
function (model;X_test=0,y_test=0,iter_points=vcat(1:1:9,10:5:99,100:50:999,1000:100:9999))
    metrics = MVHistory()
    function measuremetrics(model,iter)
        if in(iter,iter_points)
                if X_test!=0
                    y_p = model.predictproba(X_test)
                    loglike = fill!(similar(y_p),0)
                    loglike[y_test.==1] = log.(y_p[y_test.==1])
                    loglike[y_test.==-1] = log.(1.0 .-y_p[y_test.==-1])
                    push!(metrics,:test_error,iter,mean(y_test.==sign.(y_p.-0.5)))
                    push!(metrics,:mean_neg_loglikelihood,iter,mean(-loglike))
                    push!(metrics,:median_neg_log_likelihood,iter,median(-loglike))
                end
                push!(metrics,:ELBO,iter,model.elbo())
                push!(metrics,:mu,iter,model.μ)
                push!(metrics,:sigma,iter,diag(model.Σ))
                push!(metrics,:kernel_variance,iter,getvariance(model.kernel))
                push!(metrics,:kernel_param,iter,getlengthscales(model.kernel))
        end
    end #end SaveLog
    return metrics,SaveLog
end

"""
    Return a a MVHistory object and callback function for the training of multiclass classification problems
    The callback will store the ELBO and the variational parameters at every iterations included in iter_points
    If X_test and y_test are provided it will also store the test accuracy
"""
function getMultiClassLog(model;X_test=0,y_test=0,iter_points=vcat(1:1:9,10:5:99,100:50:999,1000:100:9999))
    metrics = MVHistory()
    function SaveLog(model,iter;hyper=false)
        if in(iter,iter_points)
            y_p = model.predictproba(X_test)
            accuracy = TestAccuracy(model,y_test,y_p)
            loglike = LogLikelihood(model,y_test,y_p)
            push!(metrics,:ELBO,model.elbo())
            push!(metrics,:test_error,1.0-accuracy)
            push!(metrics,:mean_neg_loglikelihood,iter,mean(-loglike))
            push!(metrics,:median_neg_loglikelihood,iter,median(-loglike))
            println("Iteration $iter : acc = $(accuracy)")
        end
    end #end SaveLog
    return metrics,SaveLog
end

"Return the Kullback Leibler divergence for a series of points given the true GPs and predicted ones"
function KLGP(mu,sig,f,sig_f)
    N = length(f)
    tot = 0.5*N*(-log.(sig_f)-1)
    tot += 0.5*sum(log.(sig)+(sig_f+(mu-f).^2)./sig)
    return tot
end

"Return the Jensen Shannon divergence for a series of points given the true GPs and predicted ones"
function JSGP(mu,sig,f,sig_f)
    N = length(f)
    tot = -N*0.25
    tot += 0.125*sum(sig./(sig_f)+(sig_f)./(sig) + (1.0./sig_f+1.0./sig).*((mu-f).^2))
end

"Return Accuracy on test set"
function TestAccuracy(model, y_test, y_predic)
    score = 0
    for i in 1:length(y_test)
        if (model.class_mapping[argmax(y_predic[i])]) == y_test[i]
            score += 1
        end
    end
    return score/length(y_test)
end
"Return the loglikelihood of the test set"
function LogLikelihood(model,y_test,y_predic)
    return [log(y_predic[i][model.ind_mapping[y_t]]) for (i,y_t) in enumerate(y_test)]
end

function plotting1D(iter,indices,X,f,sig_f,ind_points,pred_ind,X_test,pred,sig_pred,y_train,sig_train,title,lims,sequential=false)
    if sequential
        p = plot!(X[indices,1],f[indices],t=:scatter,lab="",alpha=0.6,color=:red,markerstrokewidth=0,ylims=lims)
        p = plot(X[1:(indices[1]-1),1],f[1:(indices[1]-1)],t=:scatter,lab="",color=:blue,alpha=0.4,markerstrokewidth=0,ylims=lims)
        p = plot!(X[(indices[1]+1):end,1],f[(indices[1]+1):end],t=:scatter,lab="",color=:blue,alpha=0.1,markerstrokewidth=0,ylims=lims)
    else
        p = plot(X,f,t=:scatter,lab="",color=:blue,alpha=0.1,markerstrokewidth=0,ylims=lims)
        p = plot!(X[indices,1],f[indices],t=:scatter,lab="",alpha=1.0,color=:red,markerstrokewidth=0,ylims=lims)
    end
    p = plot!(ind_points[:,1],pred_ind,t=:scatter,lab="",color=:green,ylims=lims)
    s = sortperm(X_test[:])
    p = plot!(X_test[s],pred[s]+3*sqrt.(sig_pred[s]),fill=(pred[s]-3*sqrt.(sig_pred[s])),alpha=0.3,linewidth=0,lab="",ylims=lims)
    display(plot!(X_test[s],pred[s],color=:red,lab="",title="Iteration $iter, k: $(size(ind_points,1))",ylims=lims))
    # p = plot!(X_test,pred,lab="",title="Iteration $iter, k: $(size(ind_points,1))")
    # KL = KLGP.(y_train,sig_train,f,sig_f)
    # JS = JSGP.(y_train,sig_train,f,sig_f)
    # display(plot!(twinx(),X,[KL JS],lab=["KL" "JS"]))
    return p
end

function plotting2D(iter,indices,X,f,ind_points,pred_ind,x1_test,x2_test,pred,minf,maxf,title;full=false)
    N_test = size(x1_test,1)
    p = plot(x1_test,x2_test,reshape(pred,N_test,N_test),t=:contour,clim=(minf,maxf),fill=true,lab="",title="Iteration $iter, k= $(size(ind_points,1))")
    # p = plot!(X[:,1],X[:,2],zcolor=f,t=:scatter,lab="",alpha=0.8,markerstrokewidth=0)
    if !full
        p = plot!(ind_points[:,1],ind_points[:,2],zcolor=pred_ind,t=:scatter,lab="",color=:red)
    end
    return p
end

function IntermediatePlotting(X_test,x1_test,x2_test,y_test,lims)
    return function plotevolution(model,iter)
        y_ind = model.predict(model.kmeansalg.centers)
        y_pred,sig_pred = model.predictproba(X_test)
        y_train,sig_train = model.predictproba(X_test)
        if size(X_test,2) == 1
            display(plotting1D(iter,model.MBIndices,model.X,model.y,model.noise,model.kmeansalg.centers,y_ind,X_test,y_pred,sig_pred,y_train,sig_train,model.Name,lims))
        else
            display(plotting2D(iter,model.MBIndices,model.X,model.y,model.kmeansalg.centers,y_ind,x1_test,x2_test,y_pred,minimum(model.y),maximum(model.y),model.Name))
        end
        sleep(0.05)
    end
end
end
