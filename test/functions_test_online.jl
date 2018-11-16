
function generate_random_walk_data(N,nDim,noise,monotone=true)
    X = zeros(N,nDim)
    f = zeros(N)
    df = Normal(0,0.01)
    if nDim == 1
        d = Normal(0,noise)
    else
        d = MvNormal(zeros(nDim),noise)
    end
    for i in 2:N
        if monotone
            X[i,:] = X[i-1,:].+abs.(rand(d))
        else
            X[i,:] = X[i-1,:].+rand(d)
        end
        f[i] = f[i-1]+2*rand(df)
    end
    return X,f
end

function generate_uniform_data(N,nDim,box_width)
    X = (rand(N,nDim).-0.5)*box_width
end
function generate_gaussian_data(N,nDim,variance=1.0)
    if nDim == 1
        d = Normal(0,1)
    else
        d = MvNormal(zeros(nDim),variance*Diagonal{Float64}(I,nDim))
    end
    X = rand(d,N)
end

function plotting1D(X,f,ind_points,pred_ind,X_test,pred,sig_pred,title;full=false,ylim)
    p = plot(X[:,1],f,t=:scatter,lab="",alpha=0.3,markerstrokewidth=0)
    if !full
        p = plot!(ind_points[:,1],pred_ind,t=:scatter,lab="",color=:red,ylim=ylim)
    end
    n_sig = 3
    s = sortperm(vec(X_test))
    p =  plot!(X_test[s],pred[s]+n_sig*sqrt.(sig_pred[s]),fill=(pred[s]-n_sig*sqrt.(sig_pred[s])),alpha=0.3,lab="",ylim=ylim)
    p =  plot!(X_test[s],pred[s],lab="",title=title)
    return p
end

function plotting2D(X,f,ind_points,pred_ind,x1_test,x2_test,pred,minf,maxf,title;full=false)
    N_test = size(x1_test,1)
    p = plot(x1_test,x2_test,reshape(pred,N_test,N_test),t=:contour,clim=(minf,maxf),fill=true,lab="",title=title)
    # p = plot!(X[:,1],X[:,2],zcolor=f,t=:scatter,lab="",alpha=0.8,markerstrokewidth=0)
    # p = plot!(X[:,1],X[:,2],color=:black,t=:scatter,lab="",alpha=0.6,markerstrokewidth=0)
    if !full
        p = plot!(ind_points[:,1],ind_points[:,2],zcolor=pred_ind,t=:scatter,lab="")
    end
    return p
end

function plotting_evaluation(model,iter)
    sleep(1/(iter+1))
    p = plot(model.X[:,1],model.y,t=:scatter,lab="",alpha=0.3,markerstrokewidth=0)
    n_sig = 3
    s = sortperm(vec(X_test))
    pred,sig_pred = model.predictproba(X_test)
    p =  plot!(X_test[s],pred[s]+n_sig*sqrt.(sig_pred[s]),fill=(pred[s]-n_sig*sqrt.(sig_pred[s])),alpha=0.3,lab="",ylim=lims = (-2*abs(minimum(model.y)),2*abs(maximum(model.y))))
    p =  plot!(X_test[s],pred[s],lab="",title="i : $iter, k: $(model.m)")
    p = plot!(model.X[model.MBIndices,1],model.y[model.MBIndices],color=:red,t=:scatter,lab="")
    p = plot!(model.kmeansalg.centers[:,1],model.μ,color=:black,t=:scatter,lab="")
    eval = [get_likelihood_diff(model,model.kmeansalg,model.X[i,:]) for i in model.MBIndices]
    # println(model.MBIndices)
    p = plot!(twinx(),model.X[model.MBIndices,1],eval,t=:bar, bar_width=0.02, ylim=(0,20.0),lab="")
    p = plot!(twinx(),[minimum(model.X),maximum(model.X)],[model.kmeansalg.lim,model.kmeansalg.lim],lab="",color=:red,lw=3.0,ylim=(0,20.0))
    display(p)

end

function plottingtruth(X,y,X_test,x1_test,x2_test)
    p = plot(x1_test,x2_test,reshape(y,N_test,N_test),t=:contour,fill=true,lab="",title="Truth")
    p = plot!(X[:,1],X[:,2],zcolor=y,t=:scatter,lab="",alpha=0.8,markerstrokewidth=0)
end

function RMSE(y_pred,y_test)
    return norm(y_pred-y_test)/sqrt(length(y_test))
end

function MeanTestLogLikelihood(y_pred,y_test,noise)
    return mean(logpdf.(Normal.(y_test,noise),y_pred))
end

function KLGP(mu,sig,f,sig_f)
    N = length(f)
    tot = -0.5*N
    tot += 0.5*sum(log.(sig)-log.(sig_f)+(sig_f+(mu-f).^2)./sig)
    return tot
end

function sigpartKLGP(mu,sig,f,sig_f)
    return 0.5*sum(log.(sig))
end

function sigpart2KLGP(mu,sig,f,sig_f)
    return 0.5*sum(sig_f./(sig))
end

function mupartKLGP(mu,sig,f,sig_f)
    return sum(0.5*(mu-f).^2.0./sig)
end

function JSGP(mu,sig,f,sig_f)
    N = length(f)
    tot = -N*0.25
    tot += 0.125*sum(sig./(sig_f)+(sig_f)./(sig) .+ (1.0./(sig_f)+1.0./(sig)).*((mu-f).^2))
end

function randomf(X)
    return X[:,1].^2+cos.(X[:,1]).*sin.(X[:,2])-sin.(X[:,1]).*tanh.(X[:,2])-X[:,1]
    # return X[:,1].*sin.(X[:,2])
end


function getmetrics(X_test,x1_test,x2_test,y_test,y_train,sig_train)
    nmetrics = 7
    global data = []#Matrix(undef,0,nmetrics)
    t_init = 0
    return data, function callback(model,iter)
        if iter == 2
            t_init = time_ns()
            t_diff = 0
            global data = []#Matrix(undef,0,nmetrics)
        end
        v = zeros(nmetrics)
        t_1 = time_ns()
        v[1] = (t_1-t_init)*1e-9
        y_p,sig_p = model.predictproba(X_test)
        y_t,sig_t = model.predictproba(model.X)
        v[2] = RMSE(y_p,y_test)#RMSE
        v[3] = -MeanTestLogLikelihood(y_p,y_test,noise)#MeanTestLogLikelihood
        v[4] = KLGP(y_t,sig_t,y_train,sig_train)#KL
        v[5] = JSGP(y_t,sig_t,y_train,sig_train)#JS
        v[6] = model.m #NInducing points
        v[7] = iter
        push!(data,v')
        # println(size(data))
        t_init -= time_ns()-t_1
    end
end
