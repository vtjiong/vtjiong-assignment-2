from flask import Flask, render_template, request, jsonify
import plotly
import plotly.graph_objs as go
import json
import numpy as np
import matplotlib
from datetime import datetime
matplotlib.use('Agg') 
from kmean import Kmean
from farthest import Farthest
from plusplus import Kmean_plus
from manual import Manual

app = Flask(__name__)
x =None
y=None
data=None
centers=[]
def random():
    global x
    global y
    global data
    x = np.random.uniform(-10, 10, 300)
    y = np.random.uniform(-10, 10, 300)
    data=np.column_stack((x,y))
random()
    
@app.route('/', methods=['GET', 'POST'])
def kmeans():
    global data
    if request.method=='POST':
        req_data = request.get_json()  # Get the data from the AJAX request
        action = req_data.get('action')  # Which button was pressed (step, converge, generate, reset)
        clustering=req_data.get('initMethod')
        k = req_data.get('k')  # Value of k (number of clusters)
        if action=='converge'or action=='step':
            k = int(k.strip())
        centroids=req_data.get('centroids')
        filename=None
        print(action,clustering,k,centroids)
        if action=='generate':
              random()
              new_graph=get_graph()
              return new_graph
        elif action=="converge":
            if clustering=='random':
                kmeans=Kmean(data,k)
                kmeans.lloyds()
                time=kmeans.timestamp
                filename= f'kmeans{time}.png'
            elif clustering=='farthest':
                farthest=Farthest(data,k)
                farthest.lloyds()
                time=farthest.timestamp
                filename=f'farthest{time}.png'
            elif clustering=='kmeans':
                plus=Kmean_plus(data,k)
                plus.lloyds()
                time=plus.timestamp
                filename=f'plusplus{time}.png'
            elif clustering=='manual':
                centroids=np.array(centroids)
                manual=Manual(data,k,centroids)
                manual.lloyds()
                time=manual.timestamp
                filename=f'manual{time}.png'
            if filename:
                return jsonify({'status': 'success', 'filename':filename})
            else:
                return jsonify({'status': 'error', 'message': 'Action not recognized'})
        elif action=="step":
            if clustering=='random':
                kmeans=Kmean(data,k)
                kmeans.lloyds()
                timestamp = datetime.now().strftime("%H%M%S")
                kmeans.create_gif(f'kmeans{timestamp}.gif')
                filename = f'kmeans{timestamp}.gif'
            elif clustering=='farthest':
                farthest=Farthest(data,k)
                farthest.lloyds()
                timestamp = datetime.now().strftime("%H%M%S")
                farthest.create_gif(f'farthest{timestamp}.gif')
                filename = f'farthest{timestamp}.gif'
            elif clustering=='kmeans':
                plus=Kmean_plus(data,k)
                plus.lloyds()
                timestamp=datetime.now().strftime("%H%M%S")
                plus.create_gif(f'plus{timestamp}.gif')
                filename=f'plus{timestamp}.gif'
            elif clustering=='manual':
                centroids=np.array(centroids)
                manual=Manual(data,k,centroids)
                manual.lloyds()
                timestamp=datetime.now().strftime("%H%M%S")
                manual.create_gif(f'manual{timestamp}.gif')
                filename=f'manual{timestamp}.gif'
            if filename:
                return jsonify({'status': 'success', 'filename': filename})
            else:
                return jsonify({'status': 'error', 'message': 'Action not recognized'})
        elif action=='reset':
            return get_graph()
    return render_template('index.html')


@app.route('/get_graph')
def get_graph():
    global x
    global y
    global data
    global centers
    # Create the scatter plot with data points
    scatter = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(color='blue', size=10),
    )

    data_graph = [scatter]

    # Define layout
    layout = go.Layout(
        title="Interactive KMeans Clustering",
        xaxis=dict(title="X-Axis"),
        yaxis=dict(title="Y-Axis"),
        height=600,
        showlegend=False
    )

    graph_json = json.dumps(data_graph, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify({'graph': graph_json, 'layout': layout.to_plotly_json()})

if __name__ == '__main__':
    app.run(debug=True,port=3000)