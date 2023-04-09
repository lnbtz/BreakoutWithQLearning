import tkinter as tk

import Config
from options import *


class UI:
    def __init__(self):
        self.life_switch = False
        self.env_obs_type = OPT_ENV_RAM
        self.alpha = 0
        self.epsilon = 0
        self.gamma = 0
        self.number_of_games = 0

    def start(self):
        # main window
        window = tk.Tk()
        window.title("Start Up Configuration")

        # life switch
        life_label = tk.Label(text="Life Switch")
        life_label.pack()
        life_switch_button = tk.Button(text="current state: multiple lives")
        life_switch_button.pack()

        env_obs_type = tk.Label(text="Environment Observation Type")
        env_obs_type.pack()

        def life_switch(event):
            if self.life_switch:
                self.life_switch = False
                life_switch_button.config(text="current state: one life")
            else:
                self.life_switch = True
                life_switch_button.config(text="current state: multiple lives")

        life_switch_button.bind("<Button-1>", life_switch)

        # Dropdown menu options
        options = [
            OPT_ENV_RAM,
            OPT_ENV_RGB,
            OPT_ENV_GREYSCALE
        ]

        # datatype of menu text
        clicked = tk.StringVar()

        # initial menu text
        clicked.set(OPT_ENV_RAM)

        # Create Dropdown menu
        drop = tk.OptionMenu(window, clicked, *options)
        drop.pack()

        alpha_label = tk.Label(text="Learning Rate")
        alpha_entry = tk.Entry()
        alpha_label.pack()
        alpha_entry.pack()
        epsilon_label = tk.Label(text="Exploration Rate")
        epsilon_entry = tk.Entry()
        epsilon_label.pack()
        epsilon_entry.pack()
        gamma_label = tk.Label(text="Discount Factor")
        gamma_entry = tk.Entry()
        gamma_label.pack()
        gamma_entry.pack()
        number_of_games_label = tk.Label(text="Number of Games")
        number_of_games_entry = tk.Entry()
        number_of_games_label.pack()
        number_of_games_entry.pack()
        button = tk.Button(text="submit")

        def submit(event):
            self.env_obs_type = clicked.get()
            self.alpha = float(alpha_entry.get())
            self.epsilon = float(epsilon_entry.get())
            self.gamma = float(gamma_entry.get())
            self.number_of_games = int(number_of_games_entry.get())

            window.destroy()
            start_config = Config.Config(self.life_switch, self.env_obs_type, self.alpha, self.epsilon, self.gamma,
                                         self.number_of_games)
            start_config.doRun()

        button.pack()
        button.bind("<Button-1>", submit)
        window.mainloop()
