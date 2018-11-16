using AugmentedGaussianProcesses
using Distributions, LinearAlgebra, SpecialFunctions, Random
using Plots; pyplot()

Random.seed!(42)
include("functions_test_online.jl")
noise = 0.1
k = RBFKernel(1.0)
function sample_gaussian_process(X,noise)
    N = size(X,1)
    K = AugmentedGaussianProcesses.kernelmatrix(X,kernel)+noise*Diagonal{Float64}(I,N)
    return rand(MvNormal(zeros(N),K))
end
nDim = 1
sequential = false
n = 250
N_test= 50
noise=0.01
# X,f = generate_random_walk_data(n,nDim,0.1,monotone)
X = generate_uniform_data(n,nDim,5)
# X = generate_gaussian_data(n,nDim)'
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

ps = []

#FULL GP
t_full = @elapsed global fullgp = BatchGPRegression(X,y,kernel=kernel,noise=noise)
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


k = 50
b = 10
if visual
    if nDim ==1
        plotthisshit = AugmentedGaussianProcesses.IntermediatePlotting(X_test,x1_test,x2_test,y_test,lims)
    else
        plotthisshit = AugmentedGaussianProcesses.IntermediatePlotting(X_test,x1_test,x2_test,y_test)
    end
else
    datacontain, plotthisshit = getmetrics(X_test,x1_test,x2_test,y_test,y_train,sig_train)
end



t_m = @elapsed global model = StreamingGP(X,y,Sequential=sequential,m=k,batchsize=b,verbose=0,kernel=kernel)
t_m += @elapsed model.train(iterations=iterations,callback=plotthisshit)
push!(metrics,vcat(data...))
push!(labels,"StreamingGP")
y_m, sig_m = model.predictproba(X_test)
y_indm = model.predict(model.kmeansalg.centers)
y_trainm, sig_trainm = model.predictproba(X)

if nDim == 1
    pm = plotting1D(X,y,model.kmeansalg.centers,y_indm,X_test,y_m,sig_m,"Constant lim (m=$(model.m))",ylim=lims)
    kl_m = KLGP.(y_trainm,sig_trainm,y_train,sig_train)
    kl_simple = KLGP.(y_trainm,sig_trainm,y_train,noise)
    js_m = JSGP.(y_trainm,sig_trainm,y_train,sig_train)
    js_simple = JSGP.(y_trainm,sig_trainm,y_train,noise)
    s = sortperm(vec(X))
    plot!(twinx(),X[s],[kl_m[s] js_m[s]],lab=["KL" "JS"])
elseif nDim == 2
    pm = plotting2D(X,y,model.kmeansalg.centers,y_indm,x1_test,x2_test,y_m,minf,maxf,"Constant lim (m=$(model.m))")
end
push!(ps,pm)
println("Constant Lim ($t_m s)\n\tRMSE (train) : $(RMSE(model.predict(X),y))\n\tRMSE (test) : $(RMSE(y_m,y_test))")
