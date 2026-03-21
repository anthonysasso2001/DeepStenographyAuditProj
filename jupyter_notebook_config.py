# Jupyter Notebook configuration file
# This file sets up persistent authentication for Jupyter
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

c = get_config()  # noqa

# Set a persistent token from environment variable
jupyter_token = os.getenv('JUPYTER_TOKEN', 'backup-token-9E5DEDB0656AF1D525D2C003604D0FCA0EC')
c.NotebookApp.token = jupyter_token

# Disable password authentication since we're using token
c.NotebookApp.password = ''

# Allow root user (needed for Docker)
c.NotebookApp.allow_root = True

# Bind to all interfaces
c.NotebookApp.ip = '0.0.0.0'

# Port configuration
c.NotebookApp.port = 8888

# Disable token requirement in URL when using --ip=0.0.0.0
c.NotebookApp.token_expiry = 0  # Token never expires
