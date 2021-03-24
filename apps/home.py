#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from app import app, server
import datetime
from skimage.transform import resize
import pickle
from PIL import Image
import base64
from io import BytesIO
import numpy as np

import matplotlib.pyplot as pyplot
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers

from numpy.random import randn


def generate_latent_points(latent_dim, n_samples, n_classes=10):
	x_input = randn(latent_dim * n_samples)
	z_input = x_input.reshape(n_samples, latent_dim)
	return z_input
 
def plot_generated(examples, n):
	for i in range(n * n):
		pyplot.subplot(n, n, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(examples[i, :, :])
	pyplot.savefig("/Users/ludorandon/devia/brief-23-03-avatarGenerator/plot.png") 

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Welcome to the Avatar Generator API", className="text-center")
                    , className="mb-4 mt-4")
        ]),
        dbc.Row([
            dbc.Col(html.H4(children='Avatar Generator'
                                     ))
            ]),
        dbc.Row([
            dbc.Col(html.H5(children='You can put on the button to reload new avatar')                        
                    , className="mb-4")
            ]),
        dbc.Button(
        id='click-avatar',
        children=html.Div([
            'Click Here',
            html.A(' to generated new avatar !')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '40px',
            'borderWidth': '1px',
            'borderStyle': 'solid',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'background-color':'green'
        },
        ),
        html.Div(id='output-avatar', className="mb-4"),
        html.A("Get the full code of app on my github repositary",
               href="https://github.com/Ludo-R/")
])])

@app.callback(Output('output-avatar', 'children'),
              Input('click-avatar', 'n_clicks'),)
def on_button_click(n):
    if n is None:
        return "Not clicked."
    else:
        im = None
        model = keras.models.load_model('/Users/ludorandon/devia/brief-23-03-avatarGenerator/generator.h5')
        latent_points = generate_latent_points(256, 25)
        X  = model.predict(latent_points)
        X = (X + 1) / 2.0
        pyplot.switch_backend('Agg')
        plot_generated(X, 5)

        im = Image.open("/Users/ludorandon/devia/brief-23-03-avatarGenerator/plot.png")
    
        return html.Div([
            html.Img(src=im, style={'height':'100%', 'width':'100%'}),
            html.Hr(),
            ], className="text-center")