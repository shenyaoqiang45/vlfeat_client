# vlfeat_client
vlfeat_client (dsift+GMM+SVM)

主要流程
1、Get image database

2、Train encoder and encode images
伪函数
encoder trainEncoder（Images）
{
    //获取特征集合
    descrp = vl_dsift(Images)
    
    //可选PCA projection
    
    //GMM编码
    encoder  = vl_gmm（descrp）

    //返回结果
    return encoder
}

descrp = encodeImages(encoder, Images)
{

    //获取特征集合
    descrp1= vl_dsift(Images)

   //可选PCA projection

    //投影
    descrp  = vl_fisher(encoder, descrp1)

    //返回结果
    return encoder
}

ps.主要用到vl_dsift、vl_gmm、vl_fisher


3、Train and evaluate models

descrp（encodeImages的返回值）分为测试集和训练集

descrp{train sets}送入SVM(opencv自带)进行训练，输出结果
ps.主要用到SVM(opencv自带)
