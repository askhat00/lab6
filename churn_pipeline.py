from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("ChurnPipeline").getOrCreate()

data = spark.read.csv(
    "hdfs:///user/hadoop/churn_input/Churn_Modelling.csv",
    header=True,
    inferSchema=True
)

data = data.select(
    "CreditScore","Geography","Gender","Age","Tenure",
    "Balance","NumOfProducts","EstimatedSalary","Exited"
)

geo_idx = StringIndexer(inputCol="Geography", outputCol="GeographyIndex")
gen_idx = StringIndexer(inputCol="Gender", outputCol="GenderIndex")

encoder = OneHotEncoder(
    inputCols=["GeographyIndex","GenderIndex"],
    outputCols=["GeographyVec","GenderVec"]
)

assembler = VectorAssembler(
    inputCols=[
        "CreditScore","Age","Tenure","Balance",
        "NumOfProducts","EstimatedSalary",
        "GeographyVec","GenderVec"
    ],
    outputCol="features"
)

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures"
)

lr = LogisticRegression(
    labelCol="Exited",
    featuresCol="scaledFeatures"
)

pipeline = Pipeline(stages=[
    geo_idx, gen_idx, encoder, assembler, scaler, lr
])

model = pipeline.fit(data)
predictions = model.transform(data)

evaluator = MulticlassClassificationEvaluator(
    labelCol="Exited",
    predictionCol="prediction",
    metricName="accuracy"
)

print("Accuracy:", evaluator.evaluate(predictions))
