# from flask import Flask, render_template
#
# from bokeh.embed import components
# from bokeh.plotting import figure
# from bokeh.resources import INLINE
# from bokeh.util.string import encode_utf8
#
#
# app = Flask(__name__)
#
#
# @app.route('/')
# def index():
#     return 'Hello, World!
#
# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request
import pandas as pd
from bokeh.plotting import Histogram
from bokeh.embed import components

app = Flask(__name__)

# Load the Iris Data Set
iris_df = pd.read_csv("data/iris.data",
                      names=["Sepal Length", "Sepal Width", "Petal Length",
                             "Petal Width", "Species"
                             ]
                      )

feature_names = iris_df.columns[0:-1].values.tolist()


# Create the main plot
def create_figure(current_feature_name, bins):

    p = Histogram(iris_df, current_feature_name, title=current_feature_name,
                  color='Species', bins=bins, legend='top_right', width=600,
                  height=400)

    # Set the x axis label
    p.xaxis.axis_label = current_feature_name

    # Set the y axis label
    p.yaxis.axis_label = 'Count'
    return p


# Index page
@app.route('/')
def index():
    # Determine the selected feature
    current_feature_name = request.args.get("feature_name")

    if current_feature_name is None:
        current_feature_name = "Sepal Length"

    # Create the plot
    plot = create_figure(current_feature_name, 10)

    # Embed plot into HTML via Flask Render
    script, div = components(plot)
    return render_template("iris_index1.html", script=script, div=div,
                           feature_names=feature_names,
                           current_feature_name=current_feature_name)

# With debug=True, Flask server will auto-reload
# when there are code changes
if __name__ == '__main__':
    app.run(port=5000, debug=True)
