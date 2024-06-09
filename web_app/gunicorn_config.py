from my_setup import get_inferencer


def on_starting(server):
    get_inferencer()


bind = "0.0.0.0:8000"
workers = 2
