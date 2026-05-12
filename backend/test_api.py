import sys
sys.path.append('.')
from main import get_academic_evaluation
import json

print(json.dumps(get_academic_evaluation(), indent=2))
