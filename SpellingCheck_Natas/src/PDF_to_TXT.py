from pdf2image import convert_from_path
import pytesseract
import numpy as np
import cv2
from PIL import Image
import codecs


def __binar(image):
    image = image.convert('RGB')
    npimage = np.asarray(image).astype(np.uint8)
    npimage[:, :, 0] = 0
    npimage[:, :, 2] = 0
    im = cv2.cvtColor(npimage, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(im, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binimage = Image.fromarray(thresh)
    return binimage


def get_pages_as_images(file, num_page=1):
    images = convert_from_path(file, 500, grayscale=True)
    for image in images:
        left = image.size[0]*0.05
        right = image.size[0]*0.95
        top = image.size[1]*0.05
        bottom = image.size[1]*0.95
        image = image.crop((left, top, right,bottom))
        image = __binar(image)
        #plt.imshow(image)
        #plt.show()
    return images


def get_ocr_documents(images):
    pages_text = []
    for image in images:
        pages_text.append(pytesseract.image_to_string(image, lang='por'))
    return ''.join(pages_text)


images = get_pages_as_images('../data/PDF/brascubas one page/brascubas_OP.pdf')
pages_text = get_ocr_documents(images)


with codecs.open("pages_text_original.txt", "w", "utf-8-sig") as temp:
    temp.write(pages_text)