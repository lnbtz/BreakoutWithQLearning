import tkinter as tk
from tkinter import ttk

from deepQLearning.DeepQLearning import DeepQLearning
from util import Config
from util.options import *


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
        label = label = tk.Label(
            text="Hello, Tkinter",
            fg="white",
            bg="black",
            width=10,
            height=10
        )
        label.pack()
        # life switch
        life_label = tk.Label(text="Life Switch")
        life_label.pack()
        life_switch_button = tk.Button(text="current state: multiple lives")
        life_switch_button.pack()

        env_obs_type = tk.Label(text="environment Observation Type")
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

        alpha_entry = self.add_label("Learning Rate")
        epsilon_entry = self.add_label("Exploration Rate")
        gamma_entry = self.add_label("Discount Factor")
        number_of_games_entry = self.add_label("Number of Games")
        backpropagation_rate_entry = self.add_label("Backpropagation Rate")
        replay_memory_length_entry = self.add_label("Replay Memory Length")
        min_replay_size_entry = self.add_label("Min Replay Size")
        batch_size_entry = self.add_label("Batch Size")
        copy_step_limit_entry = self.add_label("Copy Step Limit")
        max_learning_rate_entry = self.add_label("Max Learning Rate")
        min_learning_rate_entry = self.add_label("Min Learning Rate")
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
            start_config.do_run()

        button.pack()
        button.bind("<Button-1>", submit)
        window.mainloop()

    @staticmethod
    def add_label(text):
        label = tk.Label(text=text)
        label_entry = tk.Entry()
        label.pack()
        label_entry.pack()
        return label_entry
