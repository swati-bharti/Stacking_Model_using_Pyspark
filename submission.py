from pyspark.ml.feature import StringIndexer,Tokenizer,CountVectorizer
from pyspark.ml import Pipeline
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType
def base_features_gen_pipeline(input_descript_col="descript", input_category_col="category", output_feature_col="features", output_label_col="label"):
    indexer=StringIndexer(inputCol=input_category_col,outputCol=output_label_col)
    wordtokenizer=Tokenizer(inputCol=input_descript_col,outputCol="words")
    counter=CountVectorizer(inputCol="words",outputCol=output_feature_col)
    pipeline=Pipeline(stages=[indexer,wordtokenizer,counter])
    return pipeline
def gen_meta_features(training_df, nb_0, nb_1, nb_2, svm_0, svm_1, svm_2):
    training_df.cache()
    #k-fold cross validation
    for i in range(5):
        condition = training_df['group'] == i # 1
        c_train = training_df.filter(~condition).cache()
        c_test = training_df.filter(condition).cache()
        # fitting the training data into naivr bayrs and svm model
        if i == 0:
            nb_model_0 = nb_0.fit(c_train)
            nb_model_1 = nb_1.fit(c_train)
            nb_model_2 = nb_2.fit(c_train)
            svm_model_0 = svm_0.fit(c_train)
            svm_model_1 = svm_1.fit(c_train)
            svm_model_2 = svm_2.fit(c_train)


            nb_pred_0 = nb_model_0.transform(c_test)
            nb_pred_1 = nb_model_1.transform(c_test)
            nb_pred_2 = nb_model_2.transform(c_test)
            
            
            svm_pred_0 = svm_model_0.transform(c_test)
            svm_pred_1 = svm_model_1.transform(c_test)
            svm_pred_2 = svm_model_2.transform(c_test)

    #union the result of first test group with other groups
        else:
            nb_model_0 = nb_0.fit(c_train)
            nb_model_1 = nb_1.fit(c_train)
            nb_model_2 = nb_2.fit(c_train)
            svm_model_0 = svm_0.fit(c_train)
            svm_model_1 = svm_1.fit(c_train)
            svm_model_2 = svm_2.fit(c_train)


            nb_pred_0 = nb_pred_0.union(nb_model_0.transform(c_test))
            nb_pred_1 = nb_pred_1.union(nb_model_1.transform(c_test))
            nb_pred_2 = nb_pred_2.union(nb_model_2.transform(c_test))
            svm_pred_0 = svm_pred_0.union(svm_model_0.transform(c_test))
            svm_pred_1 = svm_pred_1.union(svm_model_1.transform(c_test))
            svm_pred_2 = svm_pred_2.union(svm_model_2.transform(c_test))
        
   
    
    nb_pred_0 = nb_pred_0.alias('n0')
    nb_pred_1 = nb_pred_1.alias('n1')
    nb_pred_2 = nb_pred_2.alias('n2')
    svm_pred_0 = svm_pred_0.alias('s0')
    svm_pred_1 = svm_pred_1.alias('s1')
    svm_pred_2 = svm_pred_2.alias('s2')
    
    


#joining the prediction of both the models
    df_res = nb_pred_0.join(nb_pred_1,col('n0.id') == col('n1.id'), 'left').join(nb_pred_2,col('n0.id') == col('n2.id'), 'left').join(svm_pred_0,col('n0.id') == col('s0.id'), 'left').join(svm_pred_1,col('n0.id') == col('s1.id'), 'left').join(svm_pred_2,col('n0.id') == col('s2.id'), 'left')
    
    df_res = df_res.select('n0.id','n0.features','n0.label','n0.group','n0.label_0','n0.label_1','n0.label_2','n0.nb_pred_0','n1.nb_pred_1','n2.nb_pred_2','s0.svm_pred_0','s1.svm_pred_1','s2.svm_pred_2')
    
    

    df_res = df_res.withColumn("joint_pred_0", 2*col('nb_pred_0')+col('svm_pred_0'))
    df_res = df_res.withColumn("joint_pred_1", 2*col('nb_pred_1')+col('svm_pred_1'))
    df_res = df_res.withColumn("joint_pred_2", 2*col('nb_pred_2')+col('svm_pred_2'))
    
    
    return df_res


def test_prediction(test_df, base_features_pipeline_model, gen_base_pred_pipeline_model, gen_meta_feature_pipeline_model, meta_classifier):
    test_df = base_features_pipeline_model.transform(test_df)#task 1
    test_df = gen_base_pred_pipeline_model.transform(test_df)#ex notebook
    test_df = test_df.withColumn("joint_pred_0", 2*col('nb_pred_0')+col('svm_pred_0'))
    test_df = test_df.withColumn("joint_pred_1", 2*col('nb_pred_1')+col('svm_pred_1'))
    test_df = test_df.withColumn("joint_pred_2", 2*col('nb_pred_2')+col('svm_pred_2'))
    test_df = gen_meta_feature_pipeline_model.transform(test_df)#task 2
    meta_df = meta_classifier.transform(test_df)
    meta_classifier_res = meta_df.select('id', 'label','final_prediction')
    
    
    return meta_classifier_res