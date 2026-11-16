import os,sys
from pathlib import Path
os.environ.setdefault('JWT_SECRET','test-secret-for-isolated-pytest')
os.environ.setdefault('DATABASE_URL','sqlite://')
os.environ.setdefault('REDIS_URL','redis://localhost:6379/0')
BACKEND=Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:sys.path.insert(0,str(BACKEND))
