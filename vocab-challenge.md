# 1. A link to your repo that you have shared with me.
   
   [The repo in question](https://github.com/keatontate/app_challenge_sp23/)
# 2. Explain the added value of using DataBricks in your Data Science process (using text, diagrams, or tables).
        
DataBricks takes care of the backend Apache Spark DevOps that can be time-consuming to manage for data scientists. They connect to your team's cloud provider to retrieve large datasets, and include collaborative notebook editing. This simplifies complex workflows between teams.

DataBricks also has cloud compute resources that can operate on your data, whether it's stored in another cloud provider or not. In my personal process, DataBricks is helpful for exploring large datasets I couldn't run on my own environment. I especially like the Spark optimizations and the query builder. I would like to try out its machine learning tools for tracking model performance.

# 3. Compare and contrast PySpark to either Pandas or the Tidyverse (using text, diagrams, or tables).
    
## Pandas 
- Great prototyping tool
- Runs on one machine
- Not easy to optimize for deployment
- Not as efficient for large datasets
- More straightforward to understand what's happening to the data

## PySpark
- Runs on distributed computing clusters
- Uses Spark to optimize code for extremely large datasets
- Works with more standards like ANSI SQL queries, Parquet and other data files, AWS, etc.
- Includes machine learning libraries and compatibility with other Apache services
- Runs in memory through transactions similar to a database

Overall, Pandas and PySpark are both useful tools. PySpark supports handling Pandas DataFrames as well, so it is sometimes easier to prototype something in Pandas on a local machine before deploying it to a cluster setup. I'll probably try to use PySpark more often after this class.

# 4. Explain Docker to somebody intelligent but not a tech person (using text, diagrams, or tables).
   
Docker is a way to run code in its own sandbox apart from other processes. If you've heard of virtual machines, you can think of this as virtual operating systems. It can fix dependency problems developers run into when trying to deploy their code. Building an app into a docker image allows others to run your code without worrying about installing the right dependencies. If they can run Docker, they can run your app. (As long as they have enough compute resources.)

Docker also makes it easy to deploy efficient code. Because Docker is optimized for hardware, it means that there is minimal overhead when deploying. You can base your code off of a slimmed down Python image or include an entire Linux distribution. This also allows it to scale on cloud platforms like Azure or AWS.