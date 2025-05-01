from dotenv import find_dotenv
from dotenv.main import DotEnv


API_KEYS = DotEnv(find_dotenv()).dict()

CHROMA_PATH = "chroma"
