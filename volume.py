import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from helpers import Debug, DebugMode
from model import output_label

def GetSquareSize(tw_pixel, tw_cm, th_pixel, th_cm):
    # tw_pixel      thumb width in pixels
    # tw_cm         thumb width in cm
    
    # th_pixel      thumb height in pixels
    # th_cm         thumb height in cm
    
    return 1/2 * ((tw_pixel / tw_cm) + (th_pixel / th_cm))
  
def drawGrid(image, BoxStart, BoxEnd, square):    
    fig, ax = plt.subplots()
    ax.imshow(image)
    
    Debug("Volume Grid", "Drawing Bounding Box")
    if DebugMode:
        # Only draw the bounding box if needed
        # Otherwise there is too much information on the plot
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

def CalculateArea(image, foodItem, thumbvalues, useDepth = False):    
    
    image = Image.open(image)
    image_np = np.asarray(image)
    
    thumbvalue = 18  # This is the class value in the image
    
    # These values assume the thumb is not at an angle.
    max_width_thumb = 0 # Maximum distance on the X axis (assume this is the width of the thumb)
    max_height_thumb = 0 # Maximum distance on the Y axis (assume this is the height of the thumb)
    
    food_start_x, food_start_y, food_end_x, food_end_y  = None, None, None, None
        
    allClassesInImage = {}
    
    for y in range(len(image_np)):
        width_thumb = 0
        width_food = 0
        continuous = 0 # Make sure the thumb measurement is continous

        # Scan on the X axis for the width of the thumb
        # Also get the Y axis of where the food starts
        for x in range(len(image_np[y])):
            if image_np[y][x] == 0: # Ignore background
                continue
            
            if image_np[y][x] in allClassesInImage:
                allClassesInImage[image_np[y][x]] += 1
            else:
                 allClassesInImage[image_np[y][x]] = 1
            
            if image_np[y][x] == thumbvalue:
                if continuous:
                    width_thumb += 1
                else:
                     width_thumb += 1
                     continuous = 1
            else:
                continuous = 0
                
            if image_np[y][x] in foodItem:
                width_food += 1
                food_end_y = y
                if food_start_y is None:
                    food_start_y = y
                    
        if width_thumb > max_width_thumb:
            max_width_thumb = width_thumb
            
    for x in range(len(image_np[0])):
        height_thumb = 0
        height_food = 0
        continuous = 0 # Make sure the thumb measurement is continous
        
        # Scan the Y axis for the height of the thumb
        # Also get the X axis of where the food starts
        for y in range(len(image_np)):
            if image_np[y][x] == 0: # Ignore background
                continue
            
            if image_np[y][x] == thumbvalue:
                if continuous:
                    height_thumb += 1
                else:
                    height_thumb = 1
                    continuous = 1
            else:
                 continuous = 0
                 
            if image_np[y][x] in foodItem:
                height_food += 1
                food_end_x = x
                if food_start_x is None:
                    food_start_x = x
                    
        if height_thumb > max_height_thumb:
            max_height_thumb = height_thumb
    
    Debug("Volume measurement", f"Thumb Width values: {max_width_thumb}px, {thumbvalues[0]}cm")
    
    
    if useDepth:
        square = GetSquareSize(max_width_thumb, thumbvalues[0], max_height_thumb, thumbvalues[2])
        Debug("Volume measurement", f"Thumb Height values: {max_height_thumb}px, {thumbvalues[2]}cm")
    
    else:
        square = GetSquareSize(max_width_thumb, thumbvalues[0], max_height_thumb, thumbvalues[1])
        Debug("Volume measurement", f"Thumb Height values: {max_height_thumb}px, {thumbvalues[1]}cm")
    
       
    square = round(square)
    
    if square == 0:
        raise Exception("Unable to find the thumb! Please take another picture")
    
    Debug("Volume measurement", f"Square should be '{square}px' to be equivalent to 1cm2")
    
    if not any(x in foodItem for x in allClassesInImage):
        raise Exception(f"Unable to find any of the predicted food! Found this instead: {[output_label(i - 1) for i in allClassesInImage if i != thumbvalue]}")
        
    Debug("Volume measurement", f"Food found in this image: {[output_label(i - 1) for i in allClassesInImage if i < thumbvalue]}")
    
   
    
   
    BoxStart = [food_start_x - 1, food_start_y - 1] # -1 to not draw over the food item
    BoxEnd = [food_end_x + 1, food_end_y + 1] # +1 to not draw over the food item

    drawGrid(image, BoxStart, BoxEnd, square)
    
    fullSquares = 0
    partialSquares = 0
    
    for y in range(BoxStart[1], BoxEnd[1], square):
        for x in range(BoxStart[0], BoxEnd[0], square):    
            count = 0
            for row in range(square):
                if y + row >= len(image_np):
                    break
                count += np.count_nonzero([image_np[y + row][x: x + square] == i for i in foodItem])
            percentage = count / (square * square)
            
            if percentage >= 0.6:
                fullSquares += 1
                Debug("Volume measurement", "Square {} - {:.1f}% - Full".format((x,y), percentage * 100))
            elif percentage >= 0.3 and percentage < 0.6:
                partialSquares += 1
                Debug("Volume measurement", "Square {} - {:.1f}% - Half".format((x,y), percentage * 100))
            else:
                Debug("Volume measurement", "Square {} - {:.1f}% - Ignored".format((x,y), percentage * 100))
        Debug("Volume measurement", "*" * 10)
    area = fullSquares + 1/2 * partialSquares    
    Debug("Volume measurement", f"Calculated Area: {area}")
    
    allClassesInImage.pop(thumbvalue) # Ignore thumb
    totalSum = sum(allClassesInImage.values())    

    allClassPercentages = {k: allClassesInImage.get(k) / totalSum for k in allClassesInImage}

    # Fancy print
    Debug("Volume measurement", "Class Percentages: " + ", ".join("{} - {:.1f}%".format(output_label(k-1),v * 100) for k,v in  allClassPercentages.items()))

    return area, allClassPercentages


def CalculateVolume(images, foodItem, thumbvalues):
    Debug("Volume measurement", f"Calcuating volume of: {images['top']}")
    top_image_area, percentage_top = CalculateArea(images["top"], foodItem, thumbvalues)
    
    Debug("-" * 18, "-" * 50)
    
    Debug("Volume measurement", f"Calcuating volume of: {images['side']}")
    side_image_depth, percentage_side = CalculateArea(images["side"], foodItem, thumbvalues, useDepth=True)
    
    volume = top_image_area * side_image_depth
    
    merged_dict = { k: max(percentage_top.get(k, 0), percentage_side.get(k, 0)) for k in set(percentage_top) | set(percentage_side) }

    # Fancy print
    Debug("Volume measurement", "Merged Percentages: " + ", ".join("{} - {:.1f}%".format(output_label(k-1),v * 100) for k,v in  merged_dict.items()))
    
    return volume, merged_dict