
import sys
from PIL import features

print("Python:", sys.version)
print("Pillow:", __import__("PIL").__version__)

print("\nImage rendering capabilities:")

for feature in ["raqm", "freetype2", "webp"]:
    print(f"{feature}: {features.check(feature)}")
