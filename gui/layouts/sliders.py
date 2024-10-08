import sys
import os
#pypath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
#sys.path.append(pypath)
#sys.path.append(f'{pypath}/stylegan2-ada-pytorch')

import PySimpleGUI as sg
from settings.config import Config
sg.theme(Config.gui.theme_name)

sliders = [[sg.Slider(Config.gui.shift_range, orientation="h", resolution=.01,
                      default_value=0.0, size=(30, 15), key=f"{name}"),
            sg.Text(name, auto_size_text=True)] for name in Config.gui.vector_names]
sliders += [[sg.Button("Transform", key="TRANSFORM"),
             sg.Button("Reset", key="RESET")]]