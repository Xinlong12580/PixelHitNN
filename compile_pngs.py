from PIL import Image
from pathlib import Path
import os
path = "plots/"

output_pdf = path + "/all_residuals.pdf"

# US Letter landscape, 72 DPI
page_width, page_height = 792, 612

margin = 36
gap = 18

pages = []
layers = ["L1U", "L1F","L2","L3M","L3P","L4M","L4P"]
axes = ["x","y"]
for layer in layers:
    for axe in axes:
        generic = path + f"generic_{layer}_{axe}_residuals.png"
        template = path + f"template_{layer}_{axe}_residuals.png"
        NNwithcenter =  path + f"NNwithcenter_{layer}_{axe}_residuals.png"
        NNwithgeneric =  path + f"NNwithgeneric_{layer}_{axe}_residuals.png"
        images_try = [generic, template, NNwithcenter, NNwithgeneric]
        images = [] 
        for image in images_try:
            if not os.path.exists(image):
                print(f"Waring: png files for {image} don't exist!")
                continue  
            images.append(image)
 
        page = Image.new("RGB", (page_width, page_height), "white")

        imgs = []
        for f in images:
            img = Image.open(f).convert("RGB") 
            new_size = (int(img.width // 1.3), int(img.height // 1.3))
            img = img.resize(new_size, Image.LANCZOS)
            imgs.append(img)

        page_width = (
            sum(img.width for img in imgs)
            + gap * (len(imgs) - 1)
            + len(imgs) * margin
        )
        page_height = max(img.height for img in imgs) + 2 * margin

        page = Image.new("RGB", (page_width, page_height), "white")

        x = margin
        for img in imgs:
            y = margin + (page_height - 2 * margin - img.height) // 2
            page.paste(img, (x, y))
            x += img.width + gap


        pages.append(page)

if pages:
    pages[0].save(
        output_pdf,
        save_all=True,
        append_images=pages[1:]
    )
