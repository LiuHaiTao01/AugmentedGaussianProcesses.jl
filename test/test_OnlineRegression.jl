using Distributions
using Plots
using Clustering, LinearAlgebra, Random
pyplot()
using AugmentedGaussianProcesses

Random.seed!(42)
include("functions_test_online.jl")
############ O OOO O O O O O O O O O O O
kernel = AugmentedGaussianProcesses.RBFKernel(1.0)
############ OOOOO O O O O O OOO O O O O OO O
function sample_gaussian_process(X,noise)
    N = size(X,1)
    K = AugmentedGaussianProcesses.kernelmatrix(X,kernel)+noise*Diagonal{Float64}(I,N)
    return rand(MvNormal(zeros(N),K))
end

nDim = 1
monotone = true
sequential = false
n = 1000
N_test= 50
noise=0.01
# X,f = generate_random_walk_data(n,nDim,0.1,monotone)
X = generate_uniform_data(n,nDim,5)
# X = generate_gaussian_data(n,nDim)'
X = (X.-mean(X))./sqrt(var(X))
if nDim == 1
    y = tanh.(2.0.*sample_gaussian_process(X,noise))
    mid = floor(Int64,n/2)
    ind = shuffle(1:n); ind_test = sort(ind[1:mid]); ind = sort(ind[(mid+1):end]);
    X_test = X[ind_test,:]; y_test = y[ind_test]
    X = X[ind,:]; y = y[ind]
    X_grid = range(minimum(X[:,1]),length=N_test,stop=maximum(X[:,1]))
    x1_test= X_test; x2_test =X_test
    minf=minimum(y); maxf=maximum(y)
elseif nDim == 2
    x1_test = range(minimum(X[:,1]),stop=maximum(X[:,1]),length=N_test)
    x2_test = range(minimum(X[:,2]),stop=maximum(X[:,2]),length=N_test)
    X_test = hcat([i for i in x1_test, j in x2_test][:],[j for i in x1_test, j in x2_test][:])
    y = sample_gaussian_process(vcat(X,X_test),noise)
    minf=minimum(y); maxf=maximum(y)
    y_test = y[n+1:end]
    y = y[1:n]
    # y = randomf(X)+rand(Normal(0,noise),size(X,1))
end
lims = (-2*abs(minf),2*abs(maxf))

### Non sparse GP :
ps = []

t_full = @elapsed global fullgp = AugmentedGaussianProcesses.BatchGPRegression(X,y,kernel=kernel,noise=noise)
t_full = @elapsed fullgp.train()
y_full,sig_full = fullgp.predictproba(X_test)
y_train,sig_train = fullgp.predictproba(X)
if nDim == 1
    p4 = plotting1D(X,y,[0],[0],X_test,y_full,sig_full,"Full batch GP",full=true,ylim=lims)
elseif nDim == 2
    p4 = plotting2D(X,y,[0 0],0,x1_test,x2_test,y_full,minf,maxf,"Full batch GP",full=true)
end
push!(ps,p4)
println("Full GP ($t_full s)\n\tRMSE (train) : $(RMSE(fullgp.predict(X),y))\n\tRMSE (test) : $(RMSE(y_full,y_test))")

if nDim == 1
elseif nDim == 2
end
k = 50
b = 10
dorand = false
visual = !true
if dorand
    println("Randomizing the $n points first")
    randord = shuffle(1:n)
    X = X[randord,:]
    f = f[randord]
end
if visual
    if nDim ==1
        plotthisshit = AugmentedGaussianProcesses.IntermediatePlotting(X_test,x1_test,x2_test,y_test,lims)
    else
        plotthisshit = AugmentedGaussianProcesses.IntermediatePlotting(X_test,x1_test,x2_test,y_test)
    end
else
    datacontain, plotthisshit = getmetrics(X_test,x1_test,x2_test,y_test,y_train,sig_train)
end

# kernel = AugmentedGaussianProcesses.Matern5_2Kernel(0.5)
# kernel = AugmentedGaussianProcesses.LaplaceKernel(0.5)

metrics = []
labels = []
offline = !true
webscale = !true
streaming=!true
circlek = !true
randk = true
iterations = 300
##Basic Offline KMeans
if offline
    t_off = @elapsed offgp = AugmentedGaussianProcesses.SparseGPRegression(X,y,m=50,Stochastic=true,AdaptiveLearningRate=false,τ_s=0,κ_s=0.5,Autotuning=true,OptimizeIndPoints=true,batchsize=b,verbose=0,kernel=kernel)
    plotspart = AugmentedGaussianProcesses.IntermediatePlotting(X_test,x1_test,x2_test,y_test,lims)
    t_off += @elapsed offgp.train(iterations=iterations,callback=plotthisshit)
    # t_off += @elapsed offgp.train(iterations=200,callback=plotspart)
    push!(metrics,vcat(data...))
    push!(labels,"Offline")
    y_off, sig_off = offgp.predictproba(X_test)
    y_indoff = offgp.predict(offgp.inducingPoints)
    y_trainoff, sig_trainoff = offgp.predictproba(X)
    if nDim == 1
        p1 = plotting1D(X,y,offgp.inducingPoints,y_indoff,X_test,y_off,sig_off,"Sparse GP",ylim=lims)
        kl_off = KLGP.(y_trainoff,sig_trainoff,y_train,sig_train)
        js_off = JSGP.(y_trainoff,sig_trainoff,y_train,sig_train)
        s = sortperm(vec(X))
        # plot!(twinx(),X[s],[kl_off[s] js_off[s]],lab=["KL" "JS"])
    elseif nDim == 2
        p1 = plotting2D(X,y,offgp.inducingPoints,y_indoff,x1_test,x2_test,y_off,minf,maxf,"Sparse GP")
    end
    push!(ps,p1)
    println("Offline KMeans ($t_off s)\n\tRMSE (train) : $(RMSE(offgp.predict(X),y))\n\tRMSE (test) : $(RMSE(y_off,y_test))")
end
###Online KMeans with Webscale
if webscale
    t_web = @elapsed onwebgp = AugmentedGaussianProcesses.OnlineGPRegression(X,y,kmeansalg=AugmentedGaussianProcesses.Webscale(),Sequential=false,m=k,batchsize=b,verbose=0,kernel=kernel)
    t_web = @elapsed onwebgp.train(iterations=iterations,callback=plotthisshit)
    push!(metrics,vcat(data...))
    push!(labels,"Webscale")
    y_web, sig_web = onwebgp.predictproba(X_test)
    y_indweb = onwebgp.predict(onwebgp.kmeansalg.centers)
    y_trainweb, sig_trainweb = onwebgp.predictproba(X)
    if nDim == 1
        p2 = plotting1D(X,y,onwebgp.kmeansalg.centers,y_indweb,X_test,y_web,sig_web,"Webscale KMeans (k=$(onwebgp.m))",ylim=lims)
    elseif nDim == 2
        p2 = plotting2D(X,y,onwebgp.kmeansalg.centers,y_indweb,x1_test,x2_test,y_web,minf,maxf,"Webscale KMeans (k=$(onwebgp.m))")
    end
    push!(ps,p2)
    println("Webscale KMeans ($t_web s)\n\tRMSE (train) : $(RMSE(onwebgp.predict(X),y))\n\tRMSE (test) : $(RMSE(y_web,y_test))")
end
###Online KMeans with Streaming
if streaming
    t_str = @elapsed global onstrgp = AugmentedGaussianProcesses.OnlineGPRegression(X,y,kmeansalg=AugmentedGaussianProcesses.StreamOnline(),Sequential=sequential,m=k,batchsize=b,verbose=0,kernel=kernel)
    t_str = @elapsed onstrgp.train(iterations=iterations,callback=plotthisshit)
    push!(metrics,vcat(data...))
    push!(labels,"Streaming")
    y_str,sig_str = onstrgp.predictproba(X_test)
    y_indstr = onstrgp.predict(onstrgp.kmeansalg.centers)
    y_trainstr, sig_trainstr = onstrgp.predictproba(X)

    if nDim == 1
        p3 = plotting1D(X,y,onstrgp.kmeansalg.centers,y_indstr,X_test,y_str,sig_str,"Streaming KMeans (m=$(onstrgp.m))",ylim=lims)
    elseif nDim == 2
        p3 = plotting2D(X,y,onstrgp.kmeansalg.centers,y_indstr,x1_test,x2_test,y_str,minf,maxf,"Streaming KMeans (m=$(onstrgp.m))")
    end
    push!(ps,p3)
    println("Streaming KMeans ($t_str s)\n\tRMSE (train) : $(RMSE(onstrgp.predict(X),y))\n\tRMSE (test) : $(RMSE(y_str,y_test))")
end
#### Custom K finding method with constant limit
if circlek
    t_const = @elapsed global onconstgp = AugmentedGaussianProcesses.OnlineGPRegression(X,y,kmeansalg=AugmentedGaussianProcesses.CircleKMeans(lim=0.9),Sequential=sequential,m=k,batchsize=b,verbose=0,kernel=kernel)
    t_const = @elapsed onconstgp.train(iterations=iterations,callback=plotthisshit)
    push!(metrics,vcat(data...))
    push!(labels,"CircleKMeans")
    y_const,sig_const = onconstgp.predictproba(X_test)
    y_indconst = onconstgp.predict(onconstgp.kmeansalg.centers)
    y_trainconst, sig_trainconst = onconstgp.predictproba(X)

    if nDim == 1
        pconst = plotting1D(X,y,onconstgp.kmeansalg.centers,y_indconst,X_test,y_const,sig_const,"Constant lim (m=$(onconstgp.m))",ylim=lims)
        kl_const = KLGP.(y_trainconst,sig_trainconst,y_train,sig_train)
        kl_simple = KLGP.(y_trainconst,sig_trainconst,y_train,noise)
        js_const = JSGP.(y_trainconst,sig_trainconst,y_train,sig_train)
        js_simple = JSGP.(y_trainconst,sig_trainconst,y_train,noise)
        s = sortperm(vec(X))
        plot!(twinx(),X[s],[kl_const[s] js_const[s]],lab=["KL" "JS"])
    elseif nDim == 2
        pconst = plotting2D(X,y,onconstgp.kmeansalg.centers,y_indconst,x1_test,x2_test,y_const,minf,maxf,"Constant lim (m=$(onconstgp.m))")
    end
    push!(ps,pconst)
    println("Constant Lim ($t_const s)\n\tRMSE (train) : $(RMSE(onconstgp.predict(X),y))\n\tRMSE (test) : $(RMSE(y_const,y_test))")
end
#### Custom K with random accept using both f and X
if randk
    t_rand = @elapsed onrandgp = AugmentedGaussianProcesses.OnlineGPRegression(X,y,τ_s=10,kmeansalg=AugmentedGaussianProcesses.DataSelection(func=LikelihoodImprovement,lim=10.0),Sequential=sequential,m=k,batchsize=20,verbose=0,kernel=kernel)
    # t_rand = @elapsed onrandgp = AugmentedGaussianProcesses.OnlineGPRegression(X,y,τ_s=10,kmeansalg=AugmentedGaussianProcesses.DataSelection(lim=[0.9,0.4]),Sequential=sequential,m=k,batchsize=b,verbose=0,kernel=kernel)
    # t_rand = @elapsed onrandgp.train(iterations=iterations,callback=plotting_evaluation)
    t_rand = @elapsed onrandgp.train(iterations=iterations,callback=plotthisshit)
    push!(metrics,vcat(data...))
    push!(labels,"Custom Method")
    y_rand,sig_rand = onrandgp.predictproba(X_test)
    y_indrand = onrandgp.predict(onrandgp.kmeansalg.centers)
    y_trainrand, sig_trainrand = onrandgp.predictproba(X)

    if nDim == 1
        prand = plotting1D(X,y,onrandgp.kmeansalg.centers,y_indrand,X_test,y_rand,sig_rand,"Random (m=$(onrandgp.m))",ylim=lims)
        kl_rand = KLGP.(y_trainrand,sig_trainrand,y_train,sig_train)
        js_rand = JSGP.(y_trainrand,sig_trainrand,y_train,sig_train)
        s = sortperm(vec(X))
        plot!(twinx(),X[s],[kl_rand[s] js_rand[s]],lab=["KL" "JS"])
    elseif nDim == 2
        prand = plotting2D(X,y,onrandgp.kmeansalg.centers,y_indrand,x1_test,x2_test,y_rand,minf,maxf,"Random (m=$(onrandgp.m))")
    end
    println("Rand KMeans($t_rand s)\n\tRMSE (train) : $(RMSE(onrandgp.predict(X),y))\n\tRMSE (test) : $(RMSE(y_rand,y_test))")
    push!(ps,prand)
end




# pdiv_const = plot(X,kl_const,lab="KL")
# pdiv_const = plot!(twinx(),X,js_const,lab="JS",color=:red)
# pdiv_rand = plot(X,kl_rand,lab="KL")
# pdiv_rand = plot!(twinx(),X,js_rand,lab="JS",color=:red)

if nDim == 2
    p = plottingtruth(X,y_test,X_test,x1_test,x2_test)
    push!(ps,p)
    display(plot(ps...)); gui()
else
    display(plot(ps...)); gui()
end

if !visual
    ind_x = 7 #1 for time 7 for iterations
    p_RMSE = plot(title="RMSE")
    p_LL = plot(title="Log Likelihood")
    p_KL = plot(title="KL training")
    p_JS = plot(title="JS training")
    p_m = plot(title="# ind points")
    for (i,v) in enumerate(metrics)
        plot!(p_RMSE,v[:,ind_x],v[:,2],lab=labels[i])
        plot!(p_LL,v[:,ind_x],v[:,3],lab=labels[i])
        plot!(p_KL,v[:,ind_x],v[:,4],lab=labels[i],yaxis=:log)
        plot!(p_JS,v[:,ind_x],v[:,5],lab=labels[i],yaxis=:log)
        plot!(p_m,v[:,ind_x],v[:,6],lab=labels[i])
    end
    plot!(p_RMSE,[0,maximum(vcat(metrics...)[:,ind_x])],[(RMSE(y_full,y_test)),(RMSE(y_full,y_test))],lab="Full GP",legend=:none)
    plot!(p_LL,[0,maximum(vcat(metrics...)[:,ind_x])],[(-MeanTestLogLikelihood(y_full,y_test,noise)),(-MeanTestLogLikelihood(y_full,y_test,noise))],lab="Full GP",legend=:none)
    p_metrics = plot(p_RMSE,p_LL,p_KL,p_JS,p_m)
    display(p_metrics)
end

gui()
# println("Sparsity efficiency :
    # \t Constant lim : KL: $(KLGP(y_trainconst,sig_trainconst,y_train,sig_train)), JS: $(JSGP(y_trainconst,sig_trainconst,y_train,sig_train))
    # \t Random accept : KL: $(KLGP(y_trainrand,sig_trainrand,y_train,sig_train)), JS: $(JSGP(y_trainrand,sig_trainrand,y_train,sig_train))")
# \t Offline : $(KLGP(y_trainoff,sig_trainoff,y,noise))
# \t Webscale : $(KLGP(y_trainweb,sig_trainweb,y,noise))
# \t Streaming : $(KLGP(y_trainstr,sig_trainstr,y,noise))

# model = AugmentedGaussianProcesses.BatchGPRegression(X,f,kernel=kernel)
#
#
# display(plotting2D(X,f,onstrgp.kmeansalg.centers,y_indstr,x1_test,x2_test,y_pred,"Streaming KMeans"))
# display(plotting1D(X,f,onstrgp.kmeansalg.centers,y_indstr,X_test,y_pred,"Streaming KMeans"))
