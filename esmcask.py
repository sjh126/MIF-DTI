
import os
import json
from biolmai import BioLM

# 设置 token
os.environ["BIOLMAI_TOKEN"] = "1228aef5249a765a350e6f4ad3d05798961ffeec711bdb890f108bdb20a35c4b"
response = BioLM(
    entity="esmc-600m",
    action="encode",
    params={},
    items=[
      {
        "sequence": "ACDEFGHIKL"
      }
    ]
)
print(response)