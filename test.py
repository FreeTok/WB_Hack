from PIL import Image

# Crop object form image
def crop_objects(image, detections):
    cropped_images = []
    for detection in detections:
        xmin, ymin, xmax, ymax = detection
        cropped_image = image.crop((xmin, ymin, xmax, ymax))
        cropped_images.append(cropped_image)
    return cropped_images


img = Image.open('Input/A60/image.jpg') # Load the image
crop_objects(img, [(574.1428,  172.4365,  876.4652, 1095.7454)])[0].save('new_image.jpg') # Save the image