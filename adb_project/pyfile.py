# Databricks notebook source
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, ArrayType
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, explode_outer, when, col, collect_list, size
from pyspark.sql import functions as F

container_name = "landing"
storage_account_name = "abddatabatch"
mount_point = f"/mnt/"
path = "/mnt/adb_project_movie/movie_*.json"

@F.udf(ArrayType(StringType()))
def fill_missing_genre_names(genre_ids, genre_names):
    return [genres_map.get(genre_id, "") for genre_id in genre_ids]

def mountfs(container_name, storage_account_name, mount_point):

    storage_account_access_key = dbutils.secrets.get(scope="adb_secret", key="saKey")
    if not any(mount.mountPoint == mount_point for mount in dbutils.fs.mounts()):
        dbutils.fs.mount(
            source = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/",
            mount_point = mount_point,
            extra_configs = {f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net": f"{storage_account_access_key}"}
        )
    return

genre_schema = StructType([
    StructField("id", LongType(), True),
    StructField("name", StringType(), True)
])

movie_schema = StructType([
    StructField("BackdropUrl", StringType(), True),
    StructField("Budget", DoubleType(), True),
    StructField("CreatedBy", StringType(), True),
    StructField("CreatedDate", StringType(), True),
    StructField("Id", LongType(), True),
    StructField("ImdbUrl", StringType(), True),
    StructField("OriginalLanguage", StringType(), True),
    StructField("Overview", StringType(), True),
    StructField("PosterUrl", StringType(), True),
    StructField("Price", DoubleType(), True),
    StructField("ReleaseDate", StringType(), True),
    StructField("Revenue", DoubleType(), True),
    StructField("RunTime", LongType(), True),
    StructField("Tagline", StringType(), True),
    StructField("Title", StringType(), True),
    StructField("TmdbUrl", StringType(), True),
    StructField("UpdatedBy", StringType(), True),
    StructField("UpdatedDate", StringType(), True),
    StructField("genres", ArrayType(genre_schema), True)
])

full_schema = StructType([
    StructField("movie", ArrayType(movie_schema), True)
])

spark = SparkSession.builder \
    .appName("Movie Data Analysis") \
    .getOrCreate()

raw_df = spark.read\
                .option("multiLine", True)\
                .json(path, schema= full_schema)
raw_movies_df = raw_df.select(explode("movie").alias("movie"))

movies_df = raw_movies_df.select("movie.*")

# remove dupluicated movies
movies_df = movies_df.dropDuplicates()

# create quarantined area that contains all the movies having negative runtime
quarantined_df = movies_df.filter(movies_df.RunTime < 0)

# fix minimum budget issue in Bronze
movies_df = movies_df.withColumn(
    "Budget", 
    when(movies_df.Budget < 1000000, 1000000) \
    .otherwise(movies_df.Budget)
)

movies_df = movies_df.filter(movies_df.RunTime >= 0)

exploded_genres_df = movies_df.withColumn("genre", explode_outer("genres"))

# drop duplicated genres entries and entries with genres.name == null
genres_silver_df = exploded_genres_df \
                    .withColumn("genre", explode_outer("genres")) \
                    .select("genre.id", "genre.name") \
                    .filter(col("name") != "") \
                    .distinct() \
                    .orderBy("id")

# exploded genres entry and fill in the missing genres.name

movies_silver_df = exploded_genres_df.groupBy(
    "BackdropUrl", "Budget", "CreatedBy", "CreatedDate", "Id", "ImdbUrl",
    "OriginalLanguage", "Overview", "PosterUrl", "Price", "ReleaseDate",
    "Revenue", "RunTime", "Tagline", "Title", "TmdbUrl", "UpdatedBy", "UpdatedDate"
).agg(
    collect_list("genre.id").alias("genres_id"),
    collect_list("genre.name").alias("genres_name")
)

# Fill missing genre names
genres_map = genres_silver_df.rdd.collectAsMap()

movies_silver_df = movies_silver_df.withColumn(
    "genres_name", fill_missing_genre_names("genres_id", "genres_name")
)

languages_silver_df = movies_df.select("OriginalLanguage").distinct()

# load to delta table
movies_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("/mnt/adb_p/bronze/movies/Movie")

genres_silver_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("/mnt/adb_p/silver/movies/Genres")

languages_silver_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("/mnt/adb_p/silver/movies/OriginalLanguages")

movies_silver_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("/mnt/adb_p/silver/movies/Movie")

spark.stop()