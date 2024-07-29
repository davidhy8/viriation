from flask import Flask
from config import Config
import jinja2

app = Flask(__name__)
app.jinja_env.globals.update(zip=zip)
app.config.from_object(Config)

from app import routes