"""
利用定义的高阶函数, 返回了等待原始数据和降维维度的函数
处理不同的情况

这里的函数都有两个参数, 一个是需要缩减到的维度, 一个是 df 数据框.

高阶函数固定一个参数, 等待另一个参数的变量数组

目的都一样批量处理数据

transform_to_50d=transform_to_pcadata1(50) #输入缩减维度, 等待目标数据
data_projectto_dim=transform_to_pcadata2(random_select_nrows(Xtrain))  #输入df 数据, 等待目标维度

#transform_to_2d(random_select_nrows(Xtrain))
#transform_to_3d(random_select_nrows(Xtrain))
#transform_to_50d(random_select_nrows(Xtrain))
#data_projectto_dim(2)
#data_projectto_dim(3)

reconstruct_data(imgs::DataFrame)

"""


import MLJ:transform,inverse_transform
using MLJ,DataFrames,CSV,Random,JLSO

"""
    transform_to_pcadata1(dim::Int)
输入维度, 等待图片数据

eg:  get_2pcs=transform_to_pcadata(2)
TBW
"""
function transform_to_pcadata1(dim::Int)
    mach = JLSO.load("$(pwd())/models/of-model-$(dim)pcs.jlso")[:pca]
    return (imgs::DataFrame)->begin
        @info "$dim pca proceeding..."
        pcaX = transform(mach, imgs)     # 降维数据
        # 返回降维数据
        return pcaX
    end
end


"""
    transform_to_pcadata2(imgs::DataFrame)
固定图片, 等待降维维度

eg  transform_imgs_to_n_dims()=transform_to_pcadata2(imgsArray)
TBW
"""
function transform_to_pcadata2(imgs::DataFrame)
    
    return (dim::Int)->begin
        @info "$dim pca proceeding..."
        mach = JLSO.load("$(pwd())/models/of-model-$(dim)pcs.jlso")[:pca]
        pcaX = transform(mach, imgs)     # 降维数据
        # 返回降维数据
        return pcaX
    end
end


"""
    reconstruct_data(imgs::DataFrame)
从降维数据重建图片
TBW
"""
function reconstruct_data(imgs::DataFrame)
        
        _,cols=size(imgs)
        @info "imgs  reconstructing from $(cols) dimension" 
        mach = JLSO.load("$(pwd())/models/of-model-$(cols)pcs.jlso")[:pca]
        Xr = inverse_transform(mach, imgs)  # 重建近似数据
        return Xr
end

"""
    reconstruct_data1(imgs) 高阶函数
构建imgs pca 数据, 固定 imgs array, 等待降维维度
TBW
"""
function reconstruct_data1(imgs)
    
    return (dim)->begin
        @info "$dim pca proceeding..."
        mach = JLSO.load("$(pwd())/models/of-model-$(dim)pcs.jlso")[:pca]
        Yte = transform(mach, imgs)     # 降维数据
        @info "$dim pca reconstructing..."
        Xr = inverse_transform(mach, Yte)  # 重建近似数据
        return Xr
    end
end

"""
    reconstruct_data2(dim)
固定pca 维度, 等待图片数据
TBW
"""
function reconstruct_data2(dim)
    mach = JLSO.load("$(pwd())/models/of-model-$(dim)pcs.jlso")[:pca]
    return (imgs)->begin
        @info "$dim pca proceeding..."
        Yte = transform(mach, imgs)     # 降维数据
        Xr = inverse_transform(mach, Yte)  # 重建近似数据
        return Xr
    end
end


"""
df 降维到 2d eg:df|>transform_to_2d

eg: [transform_to_2d(d) for d in [df1,df2,df3]]
"""
transform_to_2d=transform_to_pcadata1(2)

"""
df 降维到 3d eg:df|>transform_to_3d

eg: [transform_to_3d(d) for d in [df1,df2,df3]]
"""
transform_to_3d=transform_to_pcadata1(3)


export transform_to_pcadata1,transform_to_pcadata2,transform_to_2d,transform_to_3d,reconstruct_data
