

using MLJ,DataFrames,CSV,Random

 w=64
 h=64
const length=4096
const str="scikit_fetch_olivetti_faces"

function data_prepare(str)
    fetch(str) = str |> d -> CSV.File("./csv/$str.csv") |> DataFrame
    #to_ScienceType(d)=coerce(d,:label=>Multiclass)
    df = fetch(str)
    return df
end


of=olivetti_faces=data_prepare(str)
of=coerce!(of,:label=>Multiclass)
label,X=  unpack(of, ==(:label), rng=123);  


"""
    load_olivetti_faces()
    返回 olivetti face  训练数据,测试数据和标签
    return (Xtrain, Xtest), (ytrain, ytest)
    train:test=0.8
"""
function load_olivetti_faces()
    (Xtrain, Xtest), (ytrain, ytest)  = partition((X, label), 0.8, multi=true,  rng=123)
    return (Xtrain, Xtest), (ytrain, ytest)
end

return  load_olivetti_faces