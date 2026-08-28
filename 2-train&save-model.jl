
using MLJ,DataFrames,CSV,Random,JLSO

include("1-dataprocessing.jl")


(Xtrain, Xtest), (ytrain, ytest)=load_olivetti_faces()
#df=vcat(Xtrain,Xtest)


PCA = @load PCA pkg=MultivariateStats

"""
    make_model(Xtr)
    接收训练数据, 返回函数等待输入降维维度

 Arguments
- Xtr  训练数据集
 
return 新函数 arguments 为 降维维度 dim



"""
function  make_model(Xtr)
    return (dim)->begin
        model = PCA(maxoutdim=dim)
        mach = machine(model, Xtr) |> fit!
        try
            JLSO.save("$(pwd())/models/of-model-$(dim)pcs.jlso",:pca=>mach)
            @info "$(dim) dimension pca model saved"
        catch e
            @warn "$(e) has problem"
        end
    end
end

make_ol_model=make_model(Xtrain)
make_ol_model.([10,20,100,150])