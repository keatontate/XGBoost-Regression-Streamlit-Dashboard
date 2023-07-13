# API &#8614; DATA &#8614; DASHBOARD

We learned about [Streamlit](https://streamlit.io/), [Docker](https://www.docker.com/) and [machine learning](https://scikit-learn.org/stable/index.html) during our student short courses.  We will leverage all three of these in this challenge.  __We will not be using the semester project data or Spark in this challenge.__

The [data.csv](data.csv) file is the Seattle public housing data that you previously experienced in [CSE 450](https://byui-cse.github.io/cse450-course/module-03/).

## Coding Challenge

### Driving needs

_Each of the items below must be addressed by your app._

1. Allow the users to upload the [data.csv](data.csv) file into the app.
2. Require the user to select the `target` variable.
3. Allow the users to pick the features they want to use in their ML model.
4. Provide a plot that allows the users to pick and x and plot it against their target (you can have it only work for numeric values).
5. Explain your ML model (pick something fun or easy) and provide them a`fit` button.
6. Report a feature importance plot and at least one measure of model fit.
7. Provide a space on the app where the users can input new values for their features and get a prediction based on the fit model.
8. Allow them to download their [`.pickle` model file](https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/).

### Data Science Dashboard

We will use Streamlit as our prototype dashboard tool, but we need to embed that streamlit app into a Docker container.

Within this repository you can simply run `docker compose up` to leverage the `docker-compose.yaml` with your local folder synced with the container folder where the streamlit app is runnning. 

Alternatively, you could use `docker build -t streamlit .` to use the `Dockerfile` to build the image and then use `docker run -p 8501:8501 -v "$(pwd):/app:rw" streamlit` to start the container with the appropriate port and volume settings.

### Repo Structure

Your repo should be built so that I can clone the repo and run the Docker command (`docker compose up`) as described in your `readme.md` that allows me to see your app in my web browser without requiring me to install Streamlit on my computer.

1. Fork this repo to your private space
2. Add me to your private repo in your space (`hathawayj`)
3. Build your app and Docker container
4. Update your `readme.md` with details about your app and how to start it.
6. Submit the link to your repo to me in Canvas within your vocabulary/longo challenge.

## Vocabulary/Lingo Challenge

_Within a `.md` file in your repository and as a submitted `.pdf` or `.html` on Canvas, address the following items;_

1. A link to your repo that you have shared with me.
2. Explain the added value of using DataBricks in your Data Science process (using text, diagrams, or tables).
3. Compare and contrast PySpark to either Pandas or the Tidyverse (using text, diagrams, or tables).
4. Explain Docker to somebody intelligent but not a tech person (using text, diagrams, or tables).

_Your answers should be clear, detailed, and no longer than is needed. Imagine you are responding to a client or as an interview candidate._

- _Clear:_ Clean sentences and nicely laid out format.
- _Detailed:_ You touch on all the critical points of the concept. Don't speak at too high a level.
- _Brevity:_ Don't ramble. Get to the point, and don't repeat yourself.

## References

- [Streamlit Dashboard](https://streamlit.io/)
- [Docker](https://www.docker.com/)
- [Dockerfile cheat sheet](https://kapeli.com/cheat_sheets/Dockerfile.docset/Contents/Resources/Documents/index)
- [Streamlit deploy in Docker](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)
- [Streamlit and Docker](https://maelfabien.github.io/project/Streamlit/#)
- [Save and Load ML models](https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/)
- [Project 3 in CSE 450](https://byui-cse.github.io/cse450-course/module-03/)
