import sklearn
import xgboost
import altair as alt
import pandas as pd
import streamlit as st
from matplotlib import pyplot
import pickle
import numpy as np


### Defining functions up here
def plot_importances():
    ### PART 6. Report a feature importance plot and at least one measure of model fit.
    # This was very helpful: https://www.rasgoml.com/feature-engineering-tutorials/how-to-generate-feature-importance-plots-using-xgboost
    feature_importance = st.session_state.model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    fig = pyplot.figure(figsize=(12, 6))
    pyplot.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    pyplot.yticks(range(len(sorted_idx)), np.array(st.session_state.X_test.columns)[sorted_idx])
    pyplot.title('Feature Importance')
    st.pyplot(fig)

def fit_save_model():
    model = xgboost.XGBRegressor()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data[features], data[target], test_size=0.2, random_state=42)
    with st.spinner("Fitting..."):
        # Create the model and train it, use default hyperparameters for now
        model.fit(X_train, y_train)
        # for part 8 later, we pickle the model after training
    
    st.session_state.model = model
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    filename = "model.pkl"
    st.session_state.filename = filename
    pickle.dump(model, open(filename, 'wb'))

    st.write(st.session_state)

def display_metrics():
    test_predictions = st.session_state.model.predict(st.session_state.X_test)
    mse_result = sklearn.metrics.mean_squared_error(st.session_state.y_test, test_predictions, squared=False)

    st.metric("Mean Squared Error", mse_result)
    # https://www.statology.org/rmse-vs-r-squared/
    st.metric("RMSE (Root Mean Squared Error)", np.sqrt(mse_result))
    st.write("The RMSE gives the average value the target predictions are off from the actual values in units of the target.")

"""
# Regression Dashboard
This dashboard takes in a data file prepared for an XGBoost regression model. After you upload your data and select your target/features, you can download the model and run predictions on it.
"""

"""
# Upload Data
"""
### PART 1. Allow the users to upload the [data.csv](data.csv) file into the app.
# documentation - https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader
uploaded_data = st.file_uploader('Please upload your data as a .csv file.',type='csv')
if uploaded_data is not None:
    # may need to figure out how to keep in raw pickle format
    # I'll try reading it as a string first
    global data
    data = pd.read_csv(uploaded_data)
    st.dataframe(data)

    ### PART 2. Require the user to select the `target` variable.
    target = st.selectbox('Select the target variable', data.columns)
    ### PART 3. Allow the users to pick the features they want to use in their ML model.
    features = st.multiselect("Select features to use in the model. (Don't select the target)", data.columns)
    ### PART 4. Provide a plot that allows the users to pick an x and plot it against their target (you can have it only work for numeric values).
    """
    # Examine Features
    Select which feature you would like to plot against the target to view relationships. 
    """        
    feature_selection = st.radio("What feature?",(features))

    
    # I always forget Altair syntax: https://altair-viz.github.io/getting_started/overview.html#overview
    # I could probably check the columns and do box plots for discrete variables, but I'm just gonna get it working with scatter plots first.
    if st.button("Plot"):
        feature_target_chart = alt.Chart(data).mark_circle().encode(
            x=feature_selection,
            y=target
        )
        st.altair_chart(feature_target_chart, use_container_width=True)

    ### PART 5. Explain your ML model (pick something fun or easy) and provide them a`fit` button.
    """
    # Train Model
    This is an XGBoost regressor. Extreme Gradient Boosting improves the performance of other tree-like models like Scikit-learn's `RandomForest()`.
    It is a supervised machine learning model that creates a decision tree ensemble, then optimizes that ensemble to find the most efficient trees.
    
    For more information about the model, click [here](https://xgboost.readthedocs.io/en/latest/tutorials/model.html). 
    For a simpler explanation to build intuition, click [here](https://towardsdatascience.com/xgboost-regression-explain-it-to-me-like-im-10-2cf324b0bbdb). 
    
    You can click the fit button now. After the model trains it will plot the feature importances.
    If you desire to retrain the model, please refresh the page to clear the session.
    """

    if 'model' not in st.session_state:
        if st.button('Fit!'):
            fit_save_model()
        
    if 'model' in st.session_state:
        plot_importances()
        display_metrics()

        ### PART 7. Provide a space on the app where the users can input new values for their features and get a prediction based on the fit model.
        # data editor widget? https://docs.streamlit.io/library/api-reference/data/st.data_editor
        # yes, set num_rows to dynamic on the created chosen features dataframe
        """
        # Run Predictions
        Select values to test for the features you picked when building the model. The prediction will be in the same units as your target variable.
        """
        # I don't think the editor widget works. I'm going to use a form instead.

        # Need to make this a new/copied version of features otherwise it will go back to rerun the model training
        # user_prediction_data = pd.DataFrame(columns=features)
        # st.data_editor(user_prediction_data, num_rows='dynamic')

        # I'm going to follow this form code: https://subscription.packtpub.com/book/data/9781803248226/5/ch05lvl1sec31/training-models-inside-streamlit-apps
        predictions = []
        with st.form("Add Predictions"):
            for i in features:
                predictions.append(st.text_input(f"{i}"))
            submitted = st.form_submit_button()
            if submitted:
                # this was helpful to get things into the right shape for the regressor: https://machinelearningmastery.com/xgboost-for-regression/
                predictions_formatted = np.asarray([predictions], dtype=object)
                st.metric("Prediction", st.session_state.model.predict(predictions_formatted))

        ### PART 8. Allow them to download their [`.pickle` model file](https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/).
        # I used the docs as well. https://docs.streamlit.io/library/api-reference/widgets/st.download_button
        """
        # Download Model
        The model is saved in a standard pickle format. Click below to save if you're happy with the predictions. 
        If not, refresh the page and try retraining the model with different features.
        """
        with open(st.session_state.filename, "rb") as file:
            st.download_button(
                label="Download Pickle :cucumber:",
                data=file,
                file_name=st.session_state.filename
            )