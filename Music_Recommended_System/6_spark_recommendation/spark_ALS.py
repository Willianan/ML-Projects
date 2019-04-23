# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/18 16:29
@Project:Music_Recommended_System
@Filename:spark_ALS.py
"""


"""
基于spark中ALS的推荐系统
"""

import sys
import itertools
from math import sqrt
from operator import add
from os.path import join,isfile,dirname

from pyspark import SparkConf,SparkContext
from pyspark.mllib.recommendation import ALS

def parseRating(line):
	'''
	:param line: 打分格式userId::movieId::rating::timestamp
	:return: 时间戳,(userId,movieId,rating)
	'''
	fields = line.strip().split("::")
	return long(fields[3]) % 10,(int(fields[0]),int(fields[1]),float(fields[2]))

def parseMovie(line):
	'''
	:param line: 电影文件的格式movieId::movieTitle
	:return: id,title
	'''
	fields = line.strip().split("::")
	return int(fields[0]),fields[1]
def loadRatings(ratingsFile):
	"""
	:param ratingsFile: 得分文件
	:return: 评分
	"""
	if not isfile(ratingsFile):
		print("File %s does not exist" % ratingsFile)
		sys.exit(1)
	f = open(ratingsFile,'rb')
	ratings = filter(lambda r:r[2] >0,[parseRating(line)[1] for line in f])
	f.close()
	if not ratings:
		print("No ratings provided.")
		sys.exit(1)
	else:
		return ratings
def computeRmse(model,data,n):
	"""
	:param model:模型
	:param data:数据集
	:param n:
	:return:均方根误差
	"""
	predictions = model.predictAll(data.map(lambda x:(x[0],x[1])))
	predictionsAndRatings = predictions.map(lambda x:((x[0],x[1]),x[2]))\
	                                        .join(data.map(lambda x:((x[0],x[1]),x[2]))).values()
	return sqrt(predictionsAndRatings.map(lambda x:(x[0] - x[1]) ** 2).reduce(add) / float(n))

if __name__ == "__main__":
	if len(sys.argv) != 3:
		print("Usage:/path/to/spark/bin/spark-submit --driver-memory 2g" + \
		      "MovieLenALS.py movieLensDataDir personalRatingsFile")
		sys.exit(1)
	# 设定环境
	conf = SparkConf().setAppName("MovieLensALS").set("spark.executor.memory","2g")
	sc = SparkContext(conf=conf)
	# 载入打分数据
	myRatings = loadRatings(sys.argv[2])
	myRatingsRDD = sc.parallelize(myRatings,1)
	movieLensHomeDir = sys.argv[1]
	# 得到的ratings为(时间戳最后一位整数,(userId,movieId,rating))格式的RDD
	ratings = sc.textFile(join(movieLensHomeDir,"ratings.dat")).map(parseRating)
	# 得到的movie为(movieId,movieTile)格式的RDD
	movies = dict(sc.textFile(join(movieLensHomeDir,"movies.dat")).map(parseMovie).collect())
	numRatings = ratings.count()
	numUsers = ratings.values().map(lambda r:r[0]).distinct().count()
	numMovies = ratings.values().map(lambda r:r[1]).distinct().count()
	print("Got %d ratings from %d users on %d movies." % (numRatings,numUsers,numMovies))
	# 根据时间戳最后一位把整个数据集分成训练集(60%),交叉验证集(20%)和评估集(20%)
	# 训练集、交叉验证集、测试集都是(userId,movieId,rating)格式的RDD
	numPartitions = 4
	training = ratings.filter(lambda x:x[0] < 6).values().union(myRatingsRDD).repartition(numPartitions).cache()
	validation = ratings.filter(lambda x:x[0] >= 6 and x[0] < 8).values().repartition(numPartitions).cache()
	test = ratings.filter(lambda x:x[0] >= 8).values().cache()
	numTraining = training.count()
	numValidation = validation.count()
	numTest = test.count()
	print("Training: %d,Validation: %d,Test: %d" % (numTraining,numValidation,numTest))
	# 训练模型，在交叉验证集上看效果
	ranks =[8,12]
	lambdas = [0.1,10.0]
	numIters = [10,20]
	bestModel = None
	bestValidationRmse = float("inf")
	bestRank = 0
	bestLambda = -1.0
	bestNumIter = -1
	for rank,lmbda,numIter in itertools.product(ranks,lambdas,numIters):
		model = ALS.train(training,rank,numIter,lmbda)
		validationRmse = computeRmse(model,validation,numValidation)
		print("RMSE(validation) = %f for the model trained with " % validationRmse + \
		      "rank = %d, lambda = %.1f, and numIter = %d." % (rank,lmbda,numIter))
		if validationRmse < bestValidationRmse:
			bestModel = model
			bestValidationRmse = validationRmse
			bestRank = rank
			bestLambda = lmbda
			bestNumIter = numIter
	testRmse = computeRmse(bestModel,test,numIter)
	# 在测试集上评估 交叉验证集上最好的模型
	print("The best model was trained with rank = %d and lambda = %.1f, " % (bestRank,bestLambda) + \
	      "and numInter = %d, and its RMSE on the test set is %f." % (bestNumIter,testRmse))
	# 把基线模型设定为每次都返回平均得分的模型
	meanRating = training.union(validation).map(lambda x: x[2]).mean()
	baselineRmse = sqrt(test.map(lambda x:(meanRating - x[2]) ** 2).reduce(add) / numTest)
	improvement = (bestlineRmse - testRmse) / baselineRmse * 100
	print("The best model improves the baseline by %.2f%%" % (improvement))

	# 个性化的推荐
	myRatedMoviesIds = set([x[1] for x in myRatings])
	candidates = sc.parallelize([m for m in movies if m not in myRatedMoviesIds])
	predictions = bestModel.predictAll(candidates.map(lambda x:(0,x))).collect()
	recommendations = sorted(predictions,key=lambda x:x[2],reverse=True)[:50]
	print("Movies recommended for you:")
	for i in range(len(recommendations)):
		print("%2d: %s" % (i + 1,movies[recommendations[i][1]]))
	sc.stop()