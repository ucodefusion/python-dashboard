import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from geopy.geocoders import Nominatim
from datetime import datetime
import numpy as np
from geopy.exc import GeocoderTimedOut

# Load the dataset
df = pd.read_csv('SocialMediaUsersDataset2.csv', parse_dates=['DOB'])

# Sample data for geographic distribution
geo_data = df.groupby(['City', 'Country']).size().reset_index(name='EventCount')
print(geo_data)

# Initialize the geocoder
geolocator = Nominatim(user_agent="social_media_dashboard")

# Feature engineering: Calculate user engagement based on age
df['Age'] = (datetime.now() - df['DOB']).dt.days // 365
df['UserEngagement'] = df['Age'] * 2 + 10 + 5 * (df['Interests'].apply(lambda x: len(eval(x))))

# Geocode function
def geocode(city, country):
    try:
        location = geolocator.geocode(f"{city}, {country}")
        return location.latitude, location.longitude
    except GeocoderTimedOut:
        return None, None

# Custom function to split the dataset
def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]

# Simple linear regression implementation
class SimpleLinearRegression:
    def __init__(self):
        self.coefficient = 0
        self.intercept = 0

    def fit(self, X, y):
        n = len(X)
        mean_x, mean_y = np.mean(X), np.mean(y)
        SS_xy = np.sum(y * X) - n * mean_y * mean_x
        SS_xx = np.sum(X * X) - n * mean_x * mean_x
        self.coefficient = SS_xy / SS_xx
        self.intercept = mean_y - self.coefficient * mean_x

    def predict(self, X):
        return self.intercept + self.coefficient * np.array(X).flatten()

# Model training
X = df[['Age']]
y = df['UserEngagement']
X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2, random_state=42)
model = SimpleLinearRegression()
model.fit(X_train.values.flatten(), y_train)

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout of the dashboard with Bootstrap components
app.layout = dbc.Container(
    [
        dbc.Row(dbc.Col(html.H1("Social Media Users Dashboard", className="text-center mb-4"), width=12)),
        dbc.Row(
            dbc.Col(
                dcc.Dropdown(
                    id='user-dropdown',
                    options=[{'label': str(user), 'value': user} for user in df['Name']],
                    value=df['Name'].iloc[0],
                    multi=False,
                    className="w-50 mx-auto mt-4",
                )
            )
        ),
        dbc.Row(dbc.Col(html.Div(id='user-details', className="mx-auto mt-4 p-4 bg-light")), className="mb-4"),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id='gender-pie-chart', className="mx-auto mt-4"), width=6),
                dbc.Col(dcc.Graph(id='age-histogram', className="mx-auto mt-4"), width=6),
            ]
        ),
        dbc.Row(dbc.Col(dcc.Graph(id='geo-map', className="mx-auto mt-4"), width=12)),
        dbc.Row(dbc.Col(dcc.Graph(id='prediction-plot', className="mx-auto mt-4"), width=12)),
    ],
    fluid=True,
)

# Define callback to update user details, gender pie chart, age histogram, geo map, and prediction plot
@app.callback(
    [Output('user-details', 'children'),
     Output('gender-pie-chart', 'figure'),
     Output('age-histogram', 'figure'),
     Output('geo-map', 'figure'),
     Output('prediction-plot', 'figure')],
    [Input('user-dropdown', 'value')]
)
def update_user_data(selected_user):
    user_data = df[df['Name'] == selected_user].iloc[0]

    # User details
    user_details = html.Div([
        html.H2(f"Details for {user_data['Name']}"),
        html.P(f"Gender: {user_data['Gender']}"),
        html.P(f"Date of Birth: {user_data['DOB'].strftime('%Y-%m-%d')}"),
        html.P(f"City: {user_data['City']}"),
        html.P(f"Country: {user_data['Country']}"),
        html.P(f"Interests: {user_data['Interests']}"),
    ])

    # Gender distribution pie chart
    gender_counts = df['Gender'].value_counts()
    gender_pie_chart = px.pie(
        data_frame=gender_counts,
        names=gender_counts.index,
        values=gender_counts.values,
        title='Gender Distribution'
    )

    # Age histogram
    num_bins = 30
    age_histogram = px.histogram(df, x='Age', nbins=num_bins, title='Age Distribution')
    age_histogram.update_layout(bargap=0.2)

    # Prediction plot
    user_age = (datetime.now() - user_data['DOB']).days // 365
    user_engagement_prediction = model.predict([user_age])[0]
    prediction_plot = px.bar(
        x=['Actual User Engagement', 'Predicted User Engagement'],
        y=[user_data['UserEngagement'], user_engagement_prediction],
        title='User Engagement Prediction',
    )
    prediction_plot.update_layout(bargap=0.5)

    # Filtered data for geographic distribution based on interests
    filtered_df = df[df['Interests'].str.contains('book|books', case=False, na=False)]
    geo_book_event_data = filtered_df.groupby(['City', 'Country']).size().reset_index(name='EventCount')

    # Choropleth map
    choropleth_map = px.choropleth(
        geo_book_event_data,
        locations='Country', 
        locationmode='country names',
        color='EventCount',
        hover_name='City', 
        title='Geographic Distribution of Book Interests'
    )
    choropleth_map.update_geos(showcoastlines=True, coastlinecolor="Black", showland=True, landcolor="white")
    choropleth_map.update_layout(coloraxis_showscale=True, coloraxis_colorbar=dict(title="Event Count"))

    return user_details, gender_pie_chart, age_histogram, choropleth_map, prediction_plot

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
