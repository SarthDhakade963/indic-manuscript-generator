
from PIL import Image, ImageDraw, ImageFont, features

assert features.check("raqm"), "RAQM is unavailable!"

image = Image.new("RGB", (1200, 500), "#E8D6AE")
draw = ImageDraw.Draw(image)

font_path = "C:/Windows/Fonts/Nirmala.ttf"
font = ImageFont.truetype(font_path, 48, layout_engine=ImageFont.Layout.RAQM)

samples = [
    "देवनागरी परीक्षण",
    "क्ष त्र ज्ञ",
    "श्री गणेशाय नमः",
]

y = 60

for text in samples:
    draw.text((60, y), text, font=font, fill="#392719")
    y += 120

image.save("rendering_test.png")

print("Rendering test saved successfully!")
