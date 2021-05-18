import numpy as np
from PIL import Image

def GetSquareSize(tw_pixel, tw_cm, th_pixel, th_cm):
    # tw_pixel      thumb width in pixels
    # tw_cm         thumb width in cm
    
    # th_pixel      thumb height in pixels
    # th_cm         thumb height in cm
    
    return 1/2 * ((tw_pixel / tw_cm) + (th_pixel / th_cm))
  
    
def CalculateArea(image, foodItem, thumbvalues):    
    
    temp = Image.open(image)
    image = np.asarray(temp)
    
    thumbvalue = 18 
    
    max_width_thumb = 0
    max_height_thumb = 0 
        
    for y in range(len(image)):
        width_thumb = 0
        width_food = 0

        # Scan on the X axis for the width of the thumb
        for x in range(len(image[y])):
            if image[y][x] == thumbvalue:
                width_thumb += 1
                
            if image[y][x] == foodItem:
                width_food += 1
                    
        if width_thumb > max_width_thumb:
            max_width_thumb = width_thumb
            
    for x in range(len(image[0])):
        height_thumb = 0
        height_food = 0
        
        # Scan the Y axis for the height of the thumb
        for y in range(len(image)):
            if image[y][x] == thumbvalue:
                height_thumb += 1
                
            if image[y][x] == foodItem:
                height_food += 1
                    
        if height_thumb > max_height_thumb:
            max_height_thumb = height_thumb
    
    
    square = GetSquareSize(max_width_thumb, thumbvalues[0], max_height_thumb, thumbvalues[1])
    square = round(square)
    print("Thumb Width values:", max_width_thumb, thumbvalues[0])
    print("Thumb Height values:", max_height_thumb, thumbvalues[1])
    print(f"Square should be '{square}px' to be equivalent to 1cm2")
    
    if square == 0:
        print("Unable to find the thumb! Please take another picture")
    
   
