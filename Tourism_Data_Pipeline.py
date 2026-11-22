# Databricks notebook source

file_path = "/Volumes/workspace/default/tourism_data/world_data.csv"

df = spark.read.csv(
    file_path,
    header=True,
    inferSchema=True
)

# Display the dataset
display(df)


# COMMAND ----------

# Check total rows and columns
print("Total Rows:", df.count())
print("Total Columns:", len(df.columns))

# Check column names
print("\nColumn Names:")
print(df.columns)

# Check data types
print("\nData Schema:")
df.printSchema()

# COMMAND ----------

from pyspark.sql.functions import col, sum

# Count missing values in each column
missing_values = df.select([
    sum(col(c).isNull().cast("int")).alias(c)
    for c in df.columns
])

display(missing_values)

# COMMAND ----------

from pyspark.sql.functions import col, sum

# Missing values report
for column_name in df.columns:
    missing_count = df.filter(
        col(column_name).isNull()
    ).count()

    print(column_name, ":", missing_count)

# Duplicate records
print("Duplicate records:", df.count() - df.dropDuplicates().count())

# COMMAND ----------

from pyspark.sql.functions import col, sum

yearly_arrivals = df.groupBy("date").agg(
    sum("International tourism, number of arrivals").alias("Total_Arrivals")
).orderBy("date")

display(yearly_arrivals)

# COMMAND ----------

import matplotlib.pyplot as plt
from pyspark.sql.functions import sum

yearly_data = df.groupBy("date").agg(
    sum("International tourism, number of arrivals").alias("arrivals")
).orderBy("date")

pdf = yearly_data.toPandas()

plt.figure(figsize=(12, 6))
plt.plot(pdf["date"], pdf["arrivals"], marker="o")

plt.title("International Tourism Arrivals Over the Years")
plt.xlabel("Year")
plt.ylabel("Number of Tourist Arrivals")
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
from pyspark.sql.functions import sum, col

top_countries = df.groupBy("country").agg(
    sum("International tourism, number of arrivals").alias("arrivals")
).orderBy(col("arrivals").desc()).limit(10)

pdf = top_countries.toPandas()

plt.figure(figsize=(12, 6))
plt.bar(pdf["country"], pdf["arrivals"])

plt.title("Top 10 Countries by Tourist Arrivals")
plt.xlabel("Country")
plt.ylabel("Tourist Arrivals")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()

# COMMAND ----------

from pyspark.sql.functions import col

# Remove duplicate records
clean_df = df.dropDuplicates()

# Remove rows where country or year is missing
clean_df = clean_df.filter(
    col("country").isNotNull() & col("date").isNotNull()
)

# Convert numeric columns into proper data types
clean_df = clean_df.withColumn(
    "date", col("date").cast("int")
).withColumn(
    "International tourism, number of arrivals",
    col("International tourism, number of arrivals").cast("double")
).withColumn(
    "International tourism, number of departures",
    col("International tourism, number of departures").cast("double")
).withColumn(
    "GDP (current US$)",
    col("GDP (current US$)").cast("double")
).withColumn(
    "Population",
    col("Population").cast("double")
)

# Remove invalid years
clean_df = clean_df.filter(
    (col("date") >= 1900) & (col("date") <= 2026)
)

# Display cleaned data
display(clean_df)

# COMMAND ----------

print("Rows after cleaning:", clean_df.count())
print("Columns:", len(clean_df.columns))

# Check duplicates
print("Duplicate records:", clean_df.count() - clean_df.dropDuplicates().count())

# Check missing values
from pyspark.sql.functions import sum

missing_data = clean_df.select(
    *[
        sum(col(c).isNull().cast("int")).alias(c)
        for c in clean_df.columns
    ]
)

display(missing_data)

# COMMAND ----------

from pyspark.sql.functions import col

transformed_df = clean_df.select(
    col("country"),
    col("date").alias("year"),
    col("International tourism, number of arrivals").alias("tourist_arrivals"),
    col("International tourism, number of departures").alias("tourist_departures"),
    col("GDP (current US$)").alias("gdp"),
    col("Population").alias("population")
)

display(transformed_df)

# COMMAND ----------

transformed_df.write.format("delta").mode("overwrite").saveAsTable(
    "workspace.default.tourism_cleaned"
)

print("Cleaned data saved successfully!")

# COMMAND ----------

from pyspark.sql.functions import sum, col

country_arrivals = transformed_df.groupBy("country").agg(
    sum("tourist_arrivals").alias("total_arrivals")
).orderBy(col("total_arrivals").desc())

display(country_arrivals)

# COMMAND ----------

country_departures = transformed_df.groupBy("country").agg(
    sum("tourist_departures").alias("total_departures")
).orderBy(col("total_departures").desc())

display(country_departures)

# COMMAND ----------

yearly_arrivals = transformed_df.groupBy("year").agg(
    sum("tourist_arrivals").alias("total_arrivals")
).orderBy("year")

display(yearly_arrivals)

# COMMAND ----------

top_10_countries = country_arrivals.limit(10)

display(top_10_countries)

# COMMAND ----------

import matplotlib.pyplot as plt

pdf = yearly_arrivals.toPandas()

plt.figure(figsize=(12, 6))
plt.plot(pdf["year"], pdf["total_arrivals"], marker="o")

plt.title("Year-wise International Tourist Arrivals")
plt.xlabel("Year")
plt.ylabel("Total Tourist Arrivals")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# COMMAND ----------

pdf = top_10_countries.toPandas()

plt.figure(figsize=(12, 6))
plt.bar(pdf["country"], pdf["total_arrivals"])

plt.title("Top 10 Countries by Tourist Arrivals")
plt.xlabel("Country")
plt.ylabel("Total Arrivals")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# COMMAND ----------

pdf = country_departures.limit(10).toPandas()

plt.figure(figsize=(12, 6))
plt.bar(pdf["country"], pdf["total_departures"])

plt.title("Top 10 Countries by Tourist Departures")
plt.xlabel("Country")
plt.ylabel("Total Departures")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt

pdf = transformed_df.select(
    "gdp", "tourist_arrivals"
).dropna().toPandas()

plt.figure(figsize=(10, 6))
plt.scatter(pdf["gdp"], pdf["tourist_arrivals"])

plt.title("GDP vs Tourism Arrivals")
plt.xlabel("GDP (Current US$)")
plt.ylabel("Tourist Arrivals")
plt.tight_layout()
plt.show()

# COMMAND ----------

pdf = transformed_df.select(
    "population", "tourist_arrivals"
).dropna().toPandas()

plt.figure(figsize=(10, 6))
plt.scatter(pdf["population"], pdf["tourist_arrivals"])

plt.title("Population vs Tourism Arrivals")
plt.xlabel("Population")
plt.ylabel("Tourist Arrivals")
plt.tight_layout()
plt.show()

# COMMAND ----------

country_arrivals.write.format("delta").mode("overwrite").saveAsTable(
    "workspace.default.country_tourism_arrivals"
)

country_departures.write.format("delta").mode("overwrite").saveAsTable(
    "workspace.default.country_tourism_departures"
)

yearly_arrivals.write.format("delta").mode("overwrite").saveAsTable(
    "workspace.default.yearly_tourism_arrivals"
)

top_10_countries.write.format("delta").mode("overwrite").saveAsTable(
    "workspace.default.top_10_tourism_countries"
)

print("All analytics tables saved successfully!")