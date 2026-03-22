# Jupyter Server configuration file
# This file sets up persistent authentication for Jupyter
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

c = get_config()  # noqa

# Set a persistent token from environment variable
jupyter_token = os.getenv('JUPYTER_TOKEN', 'backup-token-9E5DEDB0656AF1D525D2C003604D0FCA0EC')
c.ServerApp.token = jupyter_token

# Disable password authentication since we're using token
c.ServerApp.password = ''

# Allow root user (needed for Docker)
c.ServerApp.allow_root = True

# Bind to all interfaces
c.ServerApp.ip = '0.0.0.0'

# Port configuration
c.ServerApp.port = 8888

# Token never expires
c.ServerApp.token_expiry = 0

# Use consistent user for token-authenticated requests (fixes ephemeral user generation)
c.ServerApp.authenticate_via_token_only = True
c.ServerApp.user = 'jupyter'

# Enable verbose logging to see connection details
c.ServerApp.log_level = 'DEBUG'
