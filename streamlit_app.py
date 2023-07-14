import sklearn
import xgboost
import altair as alt
import pandas as pd
import streamlit as st
import matplotlib
import pickle

"""
# Title
Explanation
"""

with st.echo(code_location='below'):
    """
    # Upload Data
    """
    ### PART 1. Allow the users to upload the [data.csv](data.csv) file into the app.
    # documentation - https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader
    uploaded_data = st.file_uploader('Please upload your data as a .csv file.',type='csv')
    if uploaded_data is not None:
        # may need to figure out how to keep in raw pickle format
        # I'll try reading it as a string first
        data = pd.read_csv(uploaded_data)
        st.dataframe(data)


    ### PART 2. Require the user to select the `target` variable.
    target = st.selectbox('Select the target variable', data.columns)

    ### PART 3. Allow the users to pick the features they want to use in their ML model.
    features = st.multiselect('Select features to use in the model', data.columns)

    ### PART 4. Provide a plot that allows the users to pick an x and plot it against their target (you can have it only work for numeric values).
    feature_selection = st.radio("What feature to plot against the target?",(features))

    # I always forget Altair syntax: https://altair-viz.github.io/getting_started/overview.html#overview
    # I could probably check the columns and do box plots for discrete variables, but I'm just gonna get it working with scatter plots first.
    feature_target_chart = alt.Chart(data).mark_circle().encode(
        x=feature_selection,
        y=target
    )
    st.altair_chart(feature_target_chart)

    ### PART 5. Explain your ML model (pick something fun or easy) and provide them a`fit` button.
    """
    # Train Model
    This is an XGBoost regressor. You can click the fit button now.
    """
    
    model = xgboost.XGBRegressor()
    # model = LogisticRegression()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data[features], data[target], test_size=0.2, random_state=42)

    # if st.button('Fit!'):
    #     with st.spinner("Fitting..."):
    #         # Create the model and train it, use default hyperparameters for now
    #         model.fit(X_train, y_train)
    #         # for part 8 later, we pickle the model after training
    #         filename = "model.pkl"
    #         pickle.dump(model, open(filename, 'wb'))

    if not st.button('Fit!'):
        st.stop()
    else:    
        with st.spinner("Fitting..."):
            # Create the model and train it, use default hyperparameters for now
            model.fit(X_train, y_train)
            # for part 8 later, we pickle the model after training
            filename = "model.pkl"
            pickle.dump(model, open(filename, 'wb'))

            # PART 6 stuff

    # else:
    #     st.write('Click to get started...')

    ### PART 6. Report a feature importance plot and at least one measure of model fit.
    # feat_imp_chart = alt.Chart(pd.Series(model.feature_importances_)).mark_bar()
    # st.altair_chart(feat_imp_chart)

    # fig, ax = matplotlib.pyplot.subplots()
    # ax = xgboost.plot_importance(model)
    # st.pyplot(fig)

    # I need to change this plotting. It breaks things when I update the page
    # xgboost.plot_importance(model)
    # st.pyplot(matplotlib.pyplot.show())

    test_predictions = model.predict(X_test)
    mse_result = sklearn.metrics.mean_squared_error(y_test, test_predictions, squared=False)
    st.metric("my metric", mse_result)

    ### PART 7. Provide a space on the app where the users can input new values for their features and get a prediction based on the fit model.
    # data editor widget? https://docs.streamlit.io/library/api-reference/data/st.data_editor
    # yes, set num_rows to dynamic on the created chosen features dataframe
    """
    # Run Predictions
    """
    # Need to make this a new/copied version of features otherwise it will go back to rerun the model training too early
    st.data_editor(features, num_rows='dynamic')

    ### PART 8. Allow them to download their [`.pickle` model file](https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/).
    # I used the docs as well. https://docs.streamlit.io/library/api-reference/widgets/st.download_button
    """
    # Download Model
    """
    with open(filename, "rb") as file:
        st.download_button(
            label="Download Pickle",
            data=file,
            file_name=filename
        )