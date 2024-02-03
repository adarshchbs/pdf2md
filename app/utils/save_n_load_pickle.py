"""Copyright (C) 2022 Adarsh Gupta"""
import pickle


def save_pickle(file_name, file):
    with open(file_name, "wb") as f:
        pickle.dump(file, f)


def load_pickle(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)
