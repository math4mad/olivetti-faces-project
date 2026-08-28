# olivetti  face recognition 的项目

## 1. dataprocessing.jl 
  
读取 csv 文件, 返回 返回 olivetti face  训练数据,测试数据和标签
`load_olivetti_faces` 函数返回训练,测试数据集和对应标签
```julia
(Xtrain, Xtest), (ytrain, ytest)=load_olivetti_faces()
```

## 2. train&save-model.jl

导入数据,返回一个高阶函数`make_model`
`make_model` 函数首先接受训练数据集, 然后等待 `dim` 需要缩减到的维度参数
最终训练的模型保存的对应的 jlso 文件中:`JLSO.save("$(pwd())/models/of-model-$(dim)pcs.jlso",:pca=>mach)`


```julia
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
#make_ol_model.([10,20,150])
```


## 3-transform-reconstruct-methods
在 `MLJ.jl`的 pca 方法和 `MultiVariate.jl` 的方法稍有不同, 使用的是 `transform`函数,而不是`project`方法 

主要函数为`transform_to_pcadata1`,为高阶函数
输入参数`dim` 为需要缩减到的维度
函数内部调用第2 步获得的模型
返回一个函数 等待 df 参数
内部调用`transform`函数对数据做降维处理
返回数据

```julia
function transform_to_pcadata(dim::Int)
    mach = JLSO.load("$(pwd())/models/of-model-$(dim)pcs.jlso")[:pca]
    return (imgs::DataFrame)->begin
        @info "$dim pca proceeding..."
        pcaX = transform(mach, imgs)     # 降维数据
        # 返回降维数据
        return pcaX
    end
end
```


`transform_to_pcadata2` 与`transform_to_pcadata1`  一样是高阶函数
但是参数输入的顺序不同, `transform_to_pcadata2`中先输入dataframe, 然后等待维度参数
```julia
    function transform_to_pcadata2(imgs::DataFrame)
        
        return (dim::Int)->begin
            @info "$dim pca proceeding..."
            mach = JLSO.load("$(pwd())/models/of-model-$(dim)pcs.jlso")[:pca]
            pcaX = transform(mach, imgs)     # 降维数据
            # 返回降维数据
            return pcaX
        end
    end
```

### 重建数据方法
从低维度数据恢复原始维度数据, 在`MLJ`中使用的方法是`inverse_transform`
`reconstruct_data` 方法首先从数据 dataframe 中获取维度`column`数据
从存储模型中调用训练模型
执行重建变换

```julia
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
```

## 维度缩减到 1d 的数据

以 faces 为例,如果缩减到一个维度, 获得的数据是所有图片共用的最大元素,也就是所有人面部共用的特征


