from flask import Flask, request
# from config import Config
import jinja2
from werkzeug.middleware.proxy_fix import ProxyFix
from metapub.convert import pmid2doi
from metapub import PubMedFetcher
# from flask_reverse_proxy_fix.middleware import ReverseProxyPrefixFix

app = Flask(__name__)
app.debug = True
app.jinja_env.globals.update(zip=zip, pmid2doi=pmid2doi, fetch=PubMedFetcher)
# app.config.from_object(Config)
from app import routes

# Figure redirect issues due to proxy
# app.config['SERVER_NAME'] = 'ood-arc.rcs.ucalgary.ca/rnode/cn0523/64234/proxy/5000/'
# app.config['APPLICATION_ROOT'] = '/rnode/cn0523/64234/proxy/5000/'
# app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1)