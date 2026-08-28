"""
利用定义的高阶函数, 返回了等待原始数据和降维维度的函数
处理不同的情况


"""


import MLJ: transform, inverse_transform
using MLJ, DataFrames, CSV, Random, JLSO
Random.seed!(121212)

include("3-transform-reconstruct-methods.jl")
include("1-dataprocessing.jl")
(Xtrain, _), (ytrain, _) = load_olivetti_faces()


# 随机选择 n行观测数据

function random_select_nrows(df, n=1)
    rows, _ = size(df)
    return df[rand(1:rows, n), :]
end


transform_to_50d = transform_to_pcadata1(50) #输入缩减维度, 等待目标数据
data_projectto_dim = transform_to_pcadata2(random_select_nrows(Xtrain))  #输入df 数据, 等待目标维度

#transform_to_2d(random_select_nrows(Xtrain))
#transform_to_3d(random_select_nrows(Xtrain))
#transform_to_50d(random_select_nrows(Xtrain))
#data_projectto_dim(2)
res = data_projectto_dim(3)  #数据投影到 三维空间
reconstruct_data(res)      #重建至df原始维度