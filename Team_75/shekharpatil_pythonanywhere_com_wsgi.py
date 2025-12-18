import os
import sys

# Add the base project directory to the path
project_home = '/home/ShekharPatil/irrigation_system'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Tell Django where your settings are
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'irrigation.settings')

# Load the WSGI application
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()
