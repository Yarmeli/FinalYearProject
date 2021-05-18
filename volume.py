import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    
        
    food_start_x = None
    food_start_y = None
    
    food_end_x = None
    food_end_y = None
    
    for y in range(len(image)):
        width_thumb = 0
        width_food = 0

        # Scan on the X axis for the width of the thumb
        # Also get the Y axis of where the food starts
        for x in range(len(image[y])):
            if image[y][x] == thumbvalue:
                width_thumb += 1
                
            if image[y][x] == foodItem:
                width_food += 1
                food_end_y = y
                if food_start_y is None:
                    food_start_y = y
                    
        if width_thumb > max_width_thumb:
            max_width_thumb = width_thumb
            
    for x in range(len(image[0])):
        height_thumb = 0
        height_food = 0
        
        # Scan the Y axis for the height of the thumb
        # Also get the X axis of where the food starts
        for y in range(len(image)):
            if image[y][x] == thumbvalue:
                height_thumb += 1
                
            if image[y][x] == foodItem:
                height_food += 1
                food_end_x = x
                if food_start_x is None:
                    food_start_x = x
                    
        if height_thumb > max_height_thumb:
            max_height_thumb = height_thumb
    
    
    square = GetSquareSize(max_width_thumb, thumbvalues[0], max_height_thumb, thumbvalues[1])
    square = round(square)
    print("Thumb Width values:", max_width_thumb, thumbvalues[0])
    print("Thumb Height values:", max_height_thumb, thumbvalues[1])
    print(f"Square should be '{square}px' to be equivalent to 1cm2")
    
    if square == 0:
        raise Exception("Unable to find the thumb! Please take another picture")
    
   
    BoxStart = [food_start_x - 1, food_start_y - 1] # -1 to not draw over the food item
    BoxEnd = [food_end_x + 1, food_end_y + 1] # +1 to not draw over the food item
    
    palette = torch.tensor([2 ** 25 - 1, 2 ** 13 - 1, 2 ** 4 - 1])
    colors = torch.as_tensor([i for i in range(19)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
        
    img = Image.fromarray(image)
    img.putpalette(colors)
    
    fig, ax = plt.subplots()
    ax.imshow(img)
    
    bounding_box = patches.Rectangle(
        BoxStart,
        BoxEnd[0] - BoxStart[0],
        BoxEnd[1] - BoxStart[1],
        linewidth=1, edgecolor='w', facecolor='none')
    ax.add_patch(bounding_box)
    
    for y in range(BoxStart[1], BoxEnd[1], square):
        for x in range(BoxStart[0], BoxEnd[0], square):
            box = patches.Rectangle((x,y), square, square, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(box)
    
    
    plt.show()
    
    
    fullSquares = 0
    partialSquares = 0
    
    for y in range(BoxStart[1], BoxEnd[1], square):
        for x in range(BoxStart[0], BoxEnd[0], square):    
            count = 0
            for row in range(square):
                count += np.count_nonzero([image[y + row][x: x + square] == foodItem])
            percentage = count / (square * square)
            
            
            if percentage >= 0.6:
                fullSquares += 1
                print("Full Square - Box:", (x,y))
            elif percentage >= 0.3 and percentage < 0.6:
                partialSquares += 1
                print("Partial Square - Box:", (x,y))
            else:
                print("Ignored - Box:", (x,y))
       
    area = fullSquares + 1/2 * partialSquares    
    print(f"Calculated Area: {area}")
