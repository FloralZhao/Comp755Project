import numpy as np
import cv2
import argparse
import os

def crop(image, gaze, scale=2):
    height, width, _ = image.shape
    new_height, new_width = height//scale, width//scale
    top_x, top_y = gaze[0] - new_height//2, gaze[1] - new_width//2
    top_x = min(max(0, top_x), height//2)
    top_y = min(max(0, top_y), width//2)
    new_image = image[top_x:top_x+new_height, top_y:top_y+new_width]
    new_image = cv2.resize(new_image, (299, 299))
    return new_image


def gaussian_blur(image, gaze, scale=2):
    height, width, _ = image.shape
    kernel_size = 11
    gaussian_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), cv2.BORDER_DEFAULT)
    mask = np.zeros_like(image) # 0: gaussian blur, 1: original image
    mask = cv2.circle(mask, (gaze[1], gaze[0]), height//(scale*2), (255,255,255), -1)
    new_image = image * (mask//255) + (1-mask//255)*gaussian_image
    new_image = cv2.resize(new_image, (299, 299))
    return new_image


def foveated_render(image, gaze):
    """Foveated image can simulate the real imaging on retina when people gaze at one point"""
    def generate_mask(gaze, height, width, sigma):
        x,y = gaze
        ix, iy = np.meshgrid(np.linspace(0,width-1,width), np.linspace(0,height-1,height)) # ix is width
        d2 = (iy-x)*(iy-x)+(ix-y)*(ix-y)
        mask = np.exp(-d2/2/sigma**2)
        return mask

    height, width, _ = image.shape
    kernel = [3, 7, 11]
    layer_0 = image
    layer_1 = cv2.GaussianBlur(image, (kernel[0],kernel[0]), cv2.BORDER_DEFAULT)
    layer_2 = cv2.GaussianBlur(image, (kernel[1],kernel[1]), cv2.BORDER_DEFAULT)
    layer_3 = cv2.GaussianBlur(image, (kernel[2],kernel[2]), cv2.BORDER_DEFAULT)
    mask_0 = generate_mask(gaze, height, width, height//8)[:,:,None]
    mask_1 = generate_mask(gaze, height, width, height//4)[:,:,None] - generate_mask(gaze, height, width, height//8)[:,:,None]
    mask_2 = generate_mask(gaze, height, width, height//2)[:,:,None] - generate_mask(gaze, height, width, height//4)[:,:,None]
    mask_3 = 1 - generate_mask(gaze, height, width, height//2)[:,:,None]
    new_image = (mask_0*layer_0 + mask_1*layer_1 + mask_2*layer_2 + mask_3*layer_3)
    new_image = new_image.astype(np.uint8)
    new_image = cv2.resize(new_image, (299, 299))
    return new_image


def get_gaze(name):
    x = name[name.find('x='):]
    x = x[2:x.find('_')]
    y = name[name.find('y='):]
    y = y[2:y.find('.')]
    gaze = np.array((int(y), int(x))) # change from image coord to array 
    return gaze


def main():
    root = '../dataset'
    dir_name = os.path.join(root, 'realisticrendering_extraprops')

    crop_dir = os.path.join(root, 'realisticrendering_extraprops_crop_3')
    gaussian_dir = os.path.join(root, 'realisticrendering_extraprops_gaussian_3')
    foveat_dir = os.path.join(root, 'realisticrendering_extraprops_foveat')
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)
    if not os.path.exists(gaussian_dir):
        os.makedirs(gaussian_dir)
    if not os.path.exists(foveat_dir):
        os.makedirs(foveat_dir)
        
    
    files = os.listdir(dir_name)
    for name in files:
        image_name = os.path.join(dir_name, name)
        gaze = get_gaze(image_name)
        image = cv2.imread(image_name)
        height, width, _ = image.shape

        image_cropped = crop(image, gaze, scale=3)
        image_gaussian = gaussian_blur(image, gaze, scale=3)
        #image_foveated = foveated_render(image, gaze)

        cv2.imwrite(os.path.join(crop_dir, name), image_cropped)
        cv2.imwrite(os.path.join(gaussian_dir, name), image_gaussian)
        #cv2.imwrite(os.path.join(foveat_dir, name), image_foveated)

        # cv2.imshow("image_cropped", image_cropped)
        # cv2.waitKey()
        # cv2.imshow("image_gaussian", image_gaussian)
        # cv2.waitKey()
        # cv2.imshow("image_foveated", image_foveated)
        # cv2.waitKey()    

if __name__=="__main__":
    main()
