import json

with open("690U.ipynb") as f:
    nb = json.load(f)

nb.get("metadata", {}).pop("widgets", None)

with open("690U.ipynb", "w") as f:
    json.dump(nb, f)