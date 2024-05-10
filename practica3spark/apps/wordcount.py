import pyspark 
 
if __name__ == "__main__":

  input_file = "/opt/spark-data/james-joyce-ulysses.txt" 	    #path to your input file here
  output_folder = "/opt/spark-data/out" 	                        #path to your output folder here
  
  # create Spark context with Spark configuration
  sc = pyspark.SparkContext(master='local[*]', appName="Word Count")
 
  # read in text file and split each document into words
  words = sc.textFile(input_file).flatMap(lambda line: line.split(" "))

  # count the occurrence of each word
  wordCounts = words.map(lambda word: (word, 1)).reduceByKey(lambda a,b:a +b)
 
  wordCounts.saveAsTextFile(output_folder)
  
  sc.stop()