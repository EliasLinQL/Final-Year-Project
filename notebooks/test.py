import plotly.graph_objects as go

fig = go.Figure(data=go.Scatter(y=[2, 3, 1]))
fig.write_image("test.png")
print("Test image saved successfully!")
