from qdrant_client.http.models import  Filter, FieldCondition, MatchValue, SearchParams
from qdrant_client.http import models as rest
import json
from dotenv import load_dotenv
from datetime import datetime
import re
import tiktoken
import common.Supabase_api as Supabase_api
from memory_profiler import profile
import posixpath, re
from urllib.parse import quote
from qdrant_client import QdrantClient
import os

qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

