
"""
of  face  pca 降维 然后串联 svm监督学习
"""

import MLJ: predict,predict_mode
using MLJ, DataFrames, CSV, Random,GLMakie
using CatBoost.MLJCatBoostInterface
Random.seed!(121212)

const w=h=64
include("1-dataprocessing.jl")
include("3-transform-reconstruct-methods.jl")
(Xtrain, Xtest), (ytrain, ytest) = load_olivetti_faces()
cat=ytrain|>Array|>levels
rows,cols=size(Xtrain)

# transform_to_100d=transform_to_pcadata1(100)
# pcaXtr=transform_to_100d(Xtrain)
# pcaXte=transform_to_100d(Xtest)
function pca_model_learning1(dim;model)
    transform_to_nd=transform_to_pcadata1(dim)
    pcaXtr=transform_to_nd(Xtrain)
    pcaXte=transform_to_nd(Xtest)
    mach = machine(model, pcaXtr,ytrain) |> fit!
    yhat = predict(mach,pcaXte)
    res=accuracy(yhat,ytest)
    return round(res,digits=3)
end

function pca_model_learning2(dim;model)
    transform_to_nd=transform_to_pcadata1(dim)
    pcaXtr=transform_to_nd(Xtrain)
    pcaXte=transform_to_nd(Xtest)
    mach = machine(model, pcaXtr,ytrain) |> fit!
    yhat = predict_mode(mach,pcaXte)
    res=accuracy(yhat,ytest)
    return round(res,digits=3)
end
SVC = @load SVC pkg=LIBSVM
model1 = SVC()

DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
model2=DecisionTreeClassifier() 

EvoTreeClassifier = @load EvoTreeClassifier pkg=EvoTrees
model3 = EvoTreeClassifier(max_depth=5, nbins=32, nrounds=100)


KNNClassifier = @load KNNClassifier pkg=NearestNeighborModels
model4 = KNNClassifier(weights = NearestNeighborModels.Inverse())

LDA = @load LDA pkg=MultivariateStats
model5 = LDA()

modelType= @load KernelPerceptron pkg = "BetaML"
model6= modelType()


const dimArr=[1,2,3,10,20,50,80,100,150,300,500]

svc_results=[pca_model_learning1(dim;model=model1) for dim in dimArr]
dt_results=[pca_model_learning2(dim;model=model2) for dim in dimArr]
evotrees_results=[pca_model_learning2(dim;model=model3) for dim in dimArr]
knn_results=[pca_model_learning2(dim;model=model4) for dim in dimArr]
lda_results=[pca_model_learning2(dim;model=model5) for dim in dimArr]
kp_results=[pca_model_learning2(dim;model=model6) for dim in dimArr]




fig=Figure(resolution=(800,600),xticks=dimArr)
ax=Axis(fig[1,1],title="olivetti faces pca=>classification accuracy",titlesize=26)
scatterlines!(ax,dimArr,svc_results,label="svm")
scatterlines!(ax,dimArr,dt_results,label="dt")
scatterlines!(ax,dimArr,evotrees_results,label="evotrees")
scatterlines!(ax,dimArr,knn_results,label="knn")
scatterlines!(ax,dimArr,lda_results,label="lda")
scatterlines!(ax,dimArr,kp_results,label="kp")

axislegend(ax)
#save("6-pca-classification-accuracy.png",fig)