using Distributions
using Plots
using Clustering, LinearAlgebra, Random
pyplot()
using AugmentedGaussianProcesses



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

function plotting1D(X,f,ind_points,pred_ind,X_test,pred,sig_pred,title;full=false)
    p = plot(X[:,1],f,t=:scatter,lab="",alpha=0.3,markerstrokewidth=0)
    if !full
        p = plot!(ind_points[:,1],pred_ind,t=:scatter,lab="",color=:red)
    end
    n_sig = 3
    s = sortperm(vec(X_test))
    p =  plot!(X_test[s],pred[s]+n_sig*sqrt.(sig_pred[s]),fill=(pred[s]-n_sig*sqrt.(sig_pred[s])),alpha=0.3,lab="")
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

function plottingtruth(X,y,X_test,x1_test,x2_test)
    p = plot(x1_test,x2_test,reshape(y,N_test,N_test),t=:contour,fill=true,lab="",title="Truth")
    p = plot!(X[:,1],X[:,2],zcolor=y,t=:scatter,lab="",alpha=0.8,markerstrokewidth=0)
end

function RMSE(y_pred,y_test)
    return norm(y_pred-y_test)/sqrt(length(y_test))
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

kernel = AugmentedGaussianProcesses.RBFKernel(0.5)

function sample_gaussian_process(X,noise)
    N = size(X,1)
    K = AugmentedGaussianProcesses.kernelmatrix(X,kernel)+noise*Diagonal{Float64}(I,N)
    return rand(MvNormal(zeros(N),K))
end

nDim = 2
monotone = true
sequential = false
n = 1000
N_test= 100
noise=0.001
# X,f = generate_random_walk_data(n,nDim,0.1,monotone)
X = generate_uniform_data(n,nDim,5)
# X = generate_gaussian_data(n,nDim)'
X = (X.-mean(X))./sqrt(var(X))
if nDim == 1
    y = sample_gaussian_process(X,noise)
    mid = floor(Int64,n/2)
    ind = shuffle(1:n); ind_test = sort(ind[1:mid]); ind = sort(ind[(mid+1):end]);
    X_test = X[ind_test,:]; y_test = y[ind_test]
    X = X[ind,:]; y = y[ind]
    X_grid = range(minimum(X[:,1]),length=N_test,stop=maximum(X[:,1]))
    x1_test= X_test; x2_test =X_test
elseif nDim == 2
    x1_test = range(minimum(X[:,1]),stop=maximum(X[:,1]),length=N_test)
    x2_test = range(minimum(X[:,2]),stop=maximum(X[:,2]),length=N_test)
    X_test = hcat([i for i in x1_test, j in x2_test][:],[j for i in x1_test, j in x2_test][:])
    y = sample_gaussian_process(vcat(X,X_test),noise)
    y_test = y[n+1:end]
    y = y[1:n]
    # y = randomf(X)+rand(Normal(0,noise),size(X,1))

    minf=minimum(randomf(X_test)); maxf=maximum(randomf(X_test))
end
if nDim == 1
elseif nDim == 2
end
k = 50
b = 20
dorand = false
if dorand
    println("Randomizing the $n points first")
    randord = shuffle(1:n)
    X = X[randord,:]
    f = f[randord]
end
if nDim ==1
    plotthisshit = AugmentedGaussianProcesses.IntermediatePlotting(X_test,x1_test,x2_test,y_test)
else
    plotthisshit = AugmentedGaussianProcesses.IntermediatePlotting(X_test,x1_test,x2_test,y_test)
end

# kernel = AugmentedGaussianProcesses.Matern5_2Kernel(0.5)
# kernel = AugmentedGaussianProcesses.LaplaceKernel(0.5)
##Basic Offline KMeans
# t_off = @elapsed offgp = AugmentedGaussianProcesses.SparseGPRegression(X,y,m=k,Stochastic=true,Autotuning=false,batchsize=b,verbose=0,kernel=kernel)
# t_off += @elapsed offgp.train(iterations=500)
# y_off, sig_off = offgp.predictproba(X_test)
# y_indoff = offgp.predict(offgp.inducingPoints)
# y_trainoff, sig_trainoff = offgp.predictproba(X)
# if nDim == 1
#     p1 = plotting1D(X,y,offgp.inducingPoints,y_indoff,X_test,y_off,sig_off,"Sparse GP")
# elseif nDim == 2
#     p1 = plotting2D(X,y,offgp.inducingPoints,y_indoff,x1_test,x2_test,y_off,minf,maxf,"Sparse GP")
# end
# println("Offline KMeans ($t_off s)\n\tRMSE (train) : $(RMSE(offgp.predict(X),y))\n\tRMSE (test) : $(RMSE(y_off,y_test))")
# ###Online KMeans with Webscale
# t_web = @elapsed onwebgp = AugmentedGaussianProcesses.OnlineGPRegression(X,y,kmeansalg=AugmentedGaussianProcesses.Webscale(),Sequential=false,m=k,batchsize=b,verbose=0,kernel=kernel)
# t_web = @elapsed onwebgp.train(iterations=10)#,callback=plotthisshit)
# y_web, sig_web = onwebgp.predictproba(X_test)
# y_indweb = onwebgp.predict(onwebgp.kmeansalg.centers)
# y_trainweb, sig_trainweb = onwebgp.predictproba(X)
#
# if nDim == 1
#     p2 = plotting1D(X,y,onwebgp.kmeansalg.centers,y_indweb,X_test,y_web,sig_web,"Webscale KMeans (k=$(onwebgp.m))")
# elseif nDim == 2
#     p2 = plotting2D(X,y,onwebgp.kmeansalg.centers,y_indweb,x1_test,x2_test,y_web,minf,maxf,"Webscale KMeans (k=$(onwebgp.m))")
# end
# println("Webscale KMeans ($t_web s)\n\tRMSE (train) : $(RMSE(onwebgp.predict(X),y))\n\tRMSE (test) : $(RMSE(y_web,y_test))")
#
#
###Online KMeans with Streaming
# t_str = @elapsed onstrgp = AugmentedGaussianProcesses.OnlineGPRegression(X,y,kmeansalg=AugmentedGaussianProcesses.StreamOnline(),Sequential=sequential,m=k,batchsize=b,verbose=0,kernel=kernel)
# t_str = @elapsed onstrgp.train(iterations=1000)#,callback=plotthisshit)
# y_str,sig_str = onstrgp.predictproba(X_test)
# y_indstr = onstrgp.predict(onstrgp.kmeansalg.centers)
# y_trainstr, sig_trainstr = onstrgp.predictproba(X)
#
# if nDim == 1
#     p3 = plotting1D(X,y,onstrgp.kmeansalg.centers,y_indstr,X_test,y_str,sig_str,"Streaming KMeans (m=$(onstrgp.m))")
# elseif nDim == 2
#     p3 = plotting2D(X,y,onstrgp.kmeansalg.centers,y_indstr,x1_test,x2_test,y_str,minf,maxf,"Streaming KMeans (m=$(onstrgp.m))")
# end
# println("Streaming KMeans ($t_str s)\n\tRMSE (train) : $(RMSE(onstrgp.predict(X),y))\n\tRMSE (test) : $(RMSE(y_str,y_test))")
### Non sparse GP :
t_full = @elapsed fullgp = AugmentedGaussianProcesses.BatchGPRegression(X,y,kernel=kernel,noise=noise)
t_full = @elapsed fullgp.train()
y_full,sig_full = fullgp.predictproba(X_test)
y_train,sig_train = fullgp.predictproba(X)
if nDim == 1
    p4 = plotting1D(X,y,[0],[0],X_test,y_full,sig_full,"Full batch GP",full=true)
elseif nDim == 2
    p4 = plotting2D(X,y,[0 0],0,x1_test,x2_test,y_full,minf,maxf,"Full batch GP",full=true)
end
println("Full GP ($t_full s)\n\tRMSE (train) : $(RMSE(fullgp.predict(X),y))\n\tRMSE (test) : $(RMSE(y_full,y_test))")



#### Custom K finding method with constant limit
t_const = @elapsed onconstgp = AugmentedGaussianProcesses.OnlineGPRegression(X,y,τ_s=1,kmeansalg=AugmentedGaussianProcesses.CircleKMeans(lim=0.95),Sequential=sequential,m=k,batchsize=b,verbose=0,kernel=kernel)
t_const = @elapsed onconstgp.train(iterations=50,callback=plotthisshit)
y_const,sig_const = onconstgp.predictproba(X_test)
y_indconst = onconstgp.predict(onconstgp.kmeansalg.centers)
y_trainconst, sig_trainconst = onconstgp.predictproba(X)

if nDim == 1
    pconst = plotting1D(X,y,onconstgp.kmeansalg.centers,y_indconst,X_test,y_const,sig_const,"Constant lim (m=$(onconstgp.m))")
    kl_const = KLGP.(y_trainconst,sig_trainconst,y_train,sig_train)
    kl_simple = KLGP.(y_trainconst,sig_trainconst,y_train,noise)
    js_const = JSGP.(y_trainconst,sig_trainconst,y_train,sig_train)
    js_simple = JSGP.(y_trainconst,sig_trainconst,y_train,noise)
    # plot!(twinx(),X,[kl_const kl_simple js_const js_simple],lab=["KL" "KL_S" "JS" "JS_S"])
    s = sortperm(vec(X))
    plot!(twinx(),X[s],[kl_const[s] js_const[s]],lab=["KL" "JS"])
    #plot!(X,y_trainconst+js_const,fill=(y_trainconst-js_co
elseif nDim == 2
    pconst = plotting2D(X,y,onconstgp.kmeansalg.centers,y_indconst,x1_test,x2_test,y_const,minf,maxf,"Constant lim (m=$(onconstgp.m))")
end
println("Constant Lim ($t_const s)\n\tRMSE (train) : $(RMSE(onconstgp.predict(X),y))\n\tRMSE (test) : $(RMSE(y_const,y_test))")


#### Custom K with random accept using both f and X
t_rand = @elapsed onrandgp = AugmentedGaussianProcesses.OnlineGPRegression(X,y,τ_s=1,kmeansalg=AugmentedGaussianProcesses.DataSelection(),Sequential=sequential,m=k,batchsize=b,verbose=0,kernel=kernel)
t_rand = @elapsed onrandgp.train(iterations=50,callback=plotthisshit)
y_rand,sig_rand = onrandgp.predictproba(X_test)
y_indrand = onrandgp.predict(onrandgp.kmeansalg.centers)
y_trainrand, sig_trainrand = onrandgp.predictproba(X)

if nDim == 1
    prand = plotting1D(X,y,onrandgp.kmeansalg.centers,y_indrand,X_test,y_rand,sig_rand,"Random (m=$(onrandgp.m))")
    kl_rand = KLGP.(y_trainrand,sig_trainrand,y_train,sig_train)
    js_rand = JSGP.(y_trainrand,sig_trainrand,y_train,sig_train)
    s = sortperm(vec(X))
    plot!(twinx(),X[s],[kl_rand[s] js_rand[s]],lab=["KL" "JS"])
elseif nDim == 2
    prand = plotting2D(X,y,onrandgp.kmeansalg.centers,y_indrand,x1_test,x2_test,y_rand,minf,maxf,"Random (m=$(onrandgp.m))")
end
println("Rand KMeans($t_rand s)\n\tRMSE (train) : $(RMSE(onrandgp.predict(X),y))\n\tRMSE (test) : $(RMSE(y_rand,y_test))")





# pdiv_const = plot(X,kl_const,lab="KL")
# pdiv_const = plot!(twinx(),X,js_const,lab="JS",color=:red)
# pdiv_rand = plot(X,kl_rand,lab="KL")
# pdiv_rand = plot!(twinx(),X,js_rand,lab="JS",color=:red)


if nDim == 2
    p = plottingtruth(X,y_test,X_test,x1_test,x2_test)
    display(plot(p,pconst,prand,p4)); gui()
else
    display(plot(p4,pconst,prand)); gui()
end



println("Sparsity efficiency :
    \t Constant lim : KL: $(KLGP(y_trainconst,sig_trainconst,y_train,sig_train)), JS: $(JSGP(y_trainconst,sig_trainconst,y_train,sig_train))
    \t Random accept : KL: $(KLGP(y_trainrand,sig_trainrand,y_train,sig_train)), JS: $(JSGP(y_trainrand,sig_trainrand,y_train,sig_train))")
# \t Offline : $(KLGP(y_trainoff,sig_trainoff,y,noise))
# \t Webscale : $(KLGP(y_trainweb,sig_trainweb,y,noise))
# \t Streaming : $(KLGP(y_trainstr,sig_trainstr,y,noise))

# model = AugmentedGaussianProcesses.BatchGPRegression(X,f,kernel=kernel)
#
#
# display(plotting2D(X,f,onstrgp.kmeansalg.centers,y_indstr,x1_test,x2_test,y_pred,"Streaming KMeans"))
# display(plotting1D(X,f,onstrgp.kmeansalg.centers,y_indstr,X_test,y_pred,"Streaming KMeans"))
