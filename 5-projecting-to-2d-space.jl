"""
图片数据投影到二维空间
"""

import MLJ: transform, inverse_transform
using MLJ, DataFrames, CSV, Random,GLMakie
Random.seed!(121212)

const w=h=64
include("1-dataprocessing.jl")
include("3-transform-reconstruct-methods.jl")
(Xtrain, Xtest), (ytrain, ytest) = load_olivetti_faces()
cat=ytrain|>Array|>levels
rows,cols=size(Xtrain)
# 随机选择 n行观测数据

pick20=rand(1:rows,20)
pickXtrain=Xtrain[pick20,:]
pickytrain=ytrain[pick20]

pcaData=transform_to_2d(pickXtrain)
reconstructImgs=reconstruct_data(pcaData)

pcaData3=transform_to_3d(pickXtrain)
reconstructImgs3=reconstruct_data(pcaData3)

transform_to_100d=transform_to_pcadata1(100)
pcaData100=transform_to_100d(pickXtrain)
reconstructImgs100=reconstruct_data(pcaData100)

df=vcat(reconstructImgs,reconstructImgs3,reconstructImgs100,pickXtrain)


function  plot_img(df)
    
    fig=Figure(resolution=(130*20,130*4))
    
    for i in 0:3
        for j in 1:20
            idx=i*20+j
            ax=Axis(fig[i+1,j],yreversed=true)
            img=df[idx,:]|>Array|>d->reshape(d,w,h)
            image!(ax,img)
            hidespines!(ax)
            hidedecorations!(ax)
        end
    end

    fig
    #save("./imgs/reconstruct-of-face.png",fig)
end


plot_img(df)