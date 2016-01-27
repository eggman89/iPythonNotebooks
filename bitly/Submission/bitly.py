# Databricks notebook source exported at Sun, 24 Jan 2016 20:34:24 UTC
# MAGIC %md
# MAGIC #####Details of the analysis
# MAGIC 
# MAGIC ######Setup:
# MAGIC Due to the hardware limitations of my laptop while processing large volume of data, I'm currently deploying the data on cloud server. This particular notebook is running on a cloud server instance provided by databricks.com. The data is stored/proessed from Amazon S3. 
# MAGIC 
# MAGIC The 6 JSON files provided for the analysis have been uploaded into the databricks database. They are named decodes01, decodes02 ..decodes06.
# MAGIC 
# MAGIC ######Details of cluster:
# MAGIC 1 Master Node, 4 Worker Nodes with 30 GB memory in each of them.
# MAGIC 
# MAGIC ######Nature of the analysis:
# MAGIC I would be using mainly Apache Spark dataframes and SQL context to perform the analysis, due to it's efficiency in distributed computing. Whenever required I would be switiching to pandas dataframe. I would be using the databricks.com tool to draw the graphs.
# MAGIC 
# MAGIC ######External Python Libraries Used:
# MAGIC - httpagentparser  
# MAGIC - numpy  
# MAGIC - pandas  
# MAGIC - matplotlib 
# MAGIC - iso3166 : to convert Country code

# COMMAND ----------

#imports and initial setup
import numpy as np
import pandas as pd
import httpagentparser
from iso3166 import countries
import matplotlib.pyplot as plt
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

#method to convert Spark Dataframe to Pandas DataFrame
def rows_to_def(rows):
    return pd.DataFrame(map(lambda e: e.asDict(), rows))
  
#load the 6 tables in 6 dataframes and merge into a single one at the end
df1 = sqlContext.sql("select * from decodes01")
df2 = sqlContext.sql("select * from decodes02")
df3 = sqlContext.sql("select * from decodes03")
df4 = sqlContext.sql("select * from decodes04")
df5 = sqlContext.sql("select * from decodes05")
df6 = sqlContext.sql("select * from decode06")
DFrame = df1.unionAll(df2).unionAll(df3).unionAll(df4).unionAll(df5).unionAll(df6)
DFrame.cache()


# COMMAND ----------

print "Total enteries in the dataset: " + str(DFrame.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ####1. Stats By Platform 
# MAGIC 
# MAGIC - Platforms interacting with Bitlinks the most  
# MAGIC - Most used browsers on Top 3 platforms to interact with Bitlinks
# MAGIC - Android or OS? : Which platforms does the top 20 traffic generating countries prefer

# COMMAND ----------

#code needed for section 1  

#methods to return OS, Platform or Broswser details, given a user_Agent string
def ua_det_os(user_agent):    
    try:
        details =  httpagentparser.detect(str(user_agent))
        if 'os' in details:
            return details['os']['name']
        else:
            return None
    except:
        return None

def ua_det_platform(user_agent):  
    try:    
        details =  httpagentparser.detect(str(user_agent))
        if 'platform' in details:
            if 'dist' in details:
                return details['dist']['name']
            elif 'platform' in details:
                return details['platform']['name']        
            else:
                return None
    except:
        return None

def ua_det_browser(user_agent):   
    try:
        details =  httpagentparser.detect(str(user_agent))
        if 'browser' in details:
            return details['browser']['name']
        else:
            return None
    except:
        return None

get_os = udf(lambda x:ua_det_os(x), StringType())
get_platform = udf(lambda x:ua_det_platform(x), StringType())
get_browser = udf(lambda x:ua_det_browser(x), StringType())


# COMMAND ----------

dftest = DFrame.withColumn('os',get_os(DFrame.a)).withColumn('browser',get_browser(DFrame.a)).withColumn('platform',get_platform(DFrame.a)).select('c','cy','a',"os","browser","platform","nk",'h','g','u')
dftest.cache()

# COMMAND ----------

# MAGIC %md 
# MAGIC Platforms interacting with Bitlinks the most

# COMMAND ----------

#Platforms interacting with Bitlinks the most  
dfc = dftest.groupBy("platform").count().sort("count",ascending=False).select("platform","count")
display(dfc)

# COMMAND ----------

# MAGIC %md
# MAGIC Most used browsers on Top 3 platforms to interact with Bitlinks

# COMMAND ----------

#Most used browsers on Top 3 platforms to interact with Bitlinks
df_android = dftest.groupBy("platform","browser").count().sort("count",ascending=False).select("platform","browser","count").where(dftest['platform'] == "Android")

df_iPhone = dftest.groupBy("platform","browser").count().sort("count",ascending=False).select("platform","browser","count").where(dftest['platform'] == "iPhone")

df_Windows = dftest.groupBy("platform","browser").count().sort("count",ascending=False).select("platform","browser","count").where(dftest['platform'] == "Windows")

display(df_android.unionAll(df_iPhone).unionAll(df_Windows))

# COMMAND ----------

# MAGIC %md
# MAGIC Android or OS? : Which platforms does the top 20 traffic generating countries prefer

# COMMAND ----------

#Android or OS? : Which platforms does the top 20 traffic generating countries prefer
dftest.registerTempTable('dftest')
df_top20 = dftest.groupBy('c').count().sort("count",ascending = False).limit(20)
pdf_top20  = rows_to_def(df_top20.collect())

top20_countries = pdf_top20.set_index('c' ).index.get_level_values('c')
sql_string1 = "SELECT c,platform,count(c) FROM dftest WHERE (c IN ("
sql_string3 = "')) AND (platform = 'Android' or platform = 'iPhone') GROUP BY c,platform"
sql_string2 = ""
sql_string4 = []


for i,val in enumerate(top20_countries):    
    sql_string4.append((val))    
for i in range(19):    
    sql_string2 =sql_string2 + "'" + str(sql_string4[i]) + "'," 
sql_string2 = sql_string2 + "'" +sql_string4[19]

sql_string = sql_string1 + sql_string2 +sql_string3
print sql_string
res1 = sqlContext.sql(sql_string).groupBy('c','platform').sum()
display(res1.sort('sum(_c2)', ascending = False))

# COMMAND ----------

# MAGIC %md
# MAGIC ######Conclusion
# MAGIC 
# MAGIC - From that data we find that PC/Windows is still the most widely used platform when it comes to interaction with the Bitlinks.
# MAGIC - However, the major two mobile platforms(iPhone and Android) combined have a significant lead compared to PC.
# MAGIC - Despite having many big players in Android browser space, the default Browsers (Chrome and the Android Default) browser are used for over 90% of the interaction.
# MAGIC - On PC, however things are divided. The default browser (IE) and Chrome have generates the same amount of traffic
# MAGIC - Globally, most people use Android devices to access the Bitlinks. However in 4 out of the top 5 countries, more people use iPhone over Android devices. In the next 15 countries, Android devices are used a lot more (Except CA and AU).

# COMMAND ----------

# MAGIC %md
# MAGIC ####2 .Stats By Country
# MAGIC 
# MAGIC - Coutries with maximum traffic
# MAGIC   - World Map
# MAGIC   - Box Plot
# MAGIC - Countries with maximum potential to expand traffic in
# MAGIC   - Percentage of traffic generated by the top 10 countries
# MAGIC   - Number of Internet users (per country) interacting with Bitlinks
# MAGIC     - For this result, I've used the data the number of Internet users per country.
# MAGIC     - The result has been acquired from here: https://en.wikipedia.org/wiki/List_of_countries_by_number_of_Internet_users
# MAGIC     - The actual traffic has been mapped to the top 10 countries with most internet users

# COMMAND ----------

# MAGIC %md
# MAGIC Countries with Most Internet users (Top 10)

# COMMAND ----------

#code needed for section 2
from pyspark.sql.functions import lit
def ua_get_alpha3(alpha2):   
    try:
      str_alpha2 =  str(alpha2)
      return countries.get(str_alpha2).alpha3
    except:
        return None
#registering custom User Defined Functions
get_alpha3 = udf(lambda x:ua_get_alpha3(x), StringType())

df_internetusers = sqlContext.sql("select * from InternetUsers")
df_internetusers_with_code = df_internetusers.withColumn("c3_1", lit(get_alpha3(df_internetusers.Country)))

display(df_internetusers.limit(10))
print "Countries with Most Internet users (Top 10)"

# COMMAND ----------

# MAGIC %md
# MAGIC Coutries with maximum traffic : World Map

# COMMAND ----------

#Coutries with maximum traffic : World Map
#dfc = DFrame.groupBy("c").count().sort("count",ascending=False).select("c","count")
from pyspark.sql.functions import lit
dfc = DFrame.withColumn("c3", lit(get_alpha3(DFrame.c))).select("c","c3").groupBy("c3").count().sort("count",ascending=False).select("c3","count")
display(dfc)

# COMMAND ----------

# MAGIC %md
# MAGIC Coutries with maximum traffic : Box Chart

# COMMAND ----------

#Coutries with maximum traffic : Box Chart
display(dfc)

# COMMAND ----------


from __future__ import division
from pyspark.sql.types import FloatType

country_traffic_percentage = udf(lambda x:((x)/18825762)*100, FloatType())  # 18825762 : total traffic
country_int_user_percentage = udf(lambda x:((x)/2782084270)*100, FloatType())  # 2782084270 : total internet users in the world

df_market2 =  df_internetusers_with_code.join(dfc,df_internetusers_with_code['c3_1'] == dfc['c3'] ).select("c3","count","InternetUsers")
df_market3 = df_market2.withColumn("Percentage_of_gen_traffic", country_traffic_percentage(df_market2['count']))
df_market4 = df_market3.withColumn("Percentage_of_internet_users", country_int_user_percentage(df_market2['InternetUsers'])).sort(df_market3['InternetUsers'], ascending = False)


# COMMAND ----------

# MAGIC %md
# MAGIC Countries with maximum potential to expand traffic in

# COMMAND ----------

#Countries with maximum potential
display(df_market4.limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ####Conclusion
# MAGIC 
# MAGIC - The maximum traffic is generated by USA. The number of interaction from USA alone is more than the traffic generated by next 7 countries alone.
# MAGIC - However, the mapping of total internet users(of a country) and Amount of traffic generated by that country (in percentage) reveals a different picture.
# MAGIC - The traffic generated by the biggest two countries internet user wise (China and India) is very small.Traffic difference exists in some other countries (Indonesia,Nigeria) too. The most probable cause seems to be the difference in time-zone.

# COMMAND ----------

# MAGIC %md
# MAGIC ####3. Stats on Repeat Clients
# MAGIC 
# MAGIC - Number of new cookies in the hours  
# MAGIC - Most preferred platform for repeat users
# MAGIC - Most preferred platform for new users

# COMMAND ----------

# MAGIC %md
# MAGIC Number of new cookies in the hours

# COMMAND ----------

#number of new cookies in the hours 
display(DFrame.groupBy('nk').count())

# COMMAND ----------

# MAGIC %md
# MAGIC Most preferred platform for repeat users

# COMMAND ----------

#Most preferred platform for repeat users
display(dftest.where(DFrame['nk'] == 1).groupBy('platform').count().sort('count',ascending = False))


# COMMAND ----------

# MAGIC %md
# MAGIC Most preferred platform for new users

# COMMAND ----------

#Most preferred platform for new users
display(dftest.where(DFrame['nk'] == 0).groupBy('platform').count().sort('count',ascending = False))


# COMMAND ----------

# MAGIC %md
# MAGIC ####Conclusion
# MAGIC - Repeat Users:
# MAGIC   - Mobile devices generate even higher number in case of repeat number. Windows(30%) is generating almost half as much traffic as the Mobile device( including iPad). 
# MAGIC - New Users:
# MAGIC   - This is the first stat where we see that the visitors prefer  PC (51%) over Mobile devices (39%). This is quite contradictory to the overall usage patterns and stats for repeat users, where Mobile devices were preferred. This could be because registering for a new service on mobile devices is cumbursome. 

# COMMAND ----------

# MAGIC %md
# MAGIC ####4. Stats on Domains of the visited websites/urls
# MAGIC 
# MAGIC - Most visited hostnames

# COMMAND ----------

# MAGIC %md
# MAGIC Most visited hostnames

# COMMAND ----------

#Most visited hostnames
import urlparse

def ua_get_hostname(url):   
    try:
      hostname = urlparse.urlparse(str(url)).hostname
      return hostname
    except:
        return None

#registering custom User Defined Functions
get_url = udf(lambda x:ua_get_hostname(x), StringType())

df_hostname = dftest.withColumn("hostname", get_url('u'))
df_results = df_hostname.groupBy(df_hostname['hostname']).count().sort("count",ascending = False)
display(df_results.where(df_results['hostname'] != 'null').limit(10))



# COMMAND ----------

# MAGIC %md
# MAGIC ####5.Other stats /Extra

# COMMAND ----------

# MAGIC %md
# MAGIC - Which top city prefer Chrome over IE on Windows

# COMMAND ----------

dftest.registerAsTable("dftest")
df_no_null = sqlContext.sql("select * from dftest where cy != 'null'  AND browser != 'null' AND platform = 'Windows' AND (browser = 'Chrome' or browser = 'Microsoft Internet Explorer')")
t1 = df_no_null.groupBy('cy','browser').count().sort('count',ascending = False)
pdf_df_browser = rows_to_def(t1.where(t1['count'] > 10000).collect())

display(t1.where(t1['count'] > 10000)) # to keep it simple, missing data won't be relavent in answering the question


# COMMAND ----------

# MAGIC %md
# MAGIC - Activity Timeline (per minute)

# COMMAND ----------

from pyspark.sql.types import TimestampType
from datetime import datetime
def ua_time(time1):   
    return  time.strftime("%Y-%m-%d %H:%M", time.gmtime(int(time1)))   
  

get_time = udf(lambda x:ua_time(x), StringType())
DFrame_time = DFrame.withColumn("time", get_time(DFrame.t)).select('time')
display(DFrame_time.groupBy('time').count().sort('time'))


# COMMAND ----------

# MAGIC %md
# MAGIC #####Details of the applicant  
# MAGIC Name: Snehasis Ghosh  
# MAGIC email_id : sghosh9@ncsu.edu  
# MAGIC Phone number: 9199468082

# COMMAND ----------


