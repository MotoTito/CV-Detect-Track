import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import mss
import time
import concurrent.futures
import argparse
import config
import glob

class TemplateImageData:
    ScaleValue=1
    ScaledImage=None
    Width=0
    Height=0

class MatchInfo:
    min_val, max_val, min_loc, max_loc = (0,0,0,0)
    scale = 0
    width=0
    height=0
    offSet = (0,0)
    def __init__(self, min_val, max_val, min_loc, max_loc, scale, width, height, offSet):
        self.min_val = min_val
        self.max_val = max_val
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.scale = scale
        self.width = width
        self.height = height
        self.offSet = offSet

class DetectionResult:
    detectionImg = None
    detectionValue = 0
    centerPoint = (0,0)

    def __init__(self, detectionImg = None, detectionValue = 0, centerPoint = (0,0)) -> None:
        self.detectionImg = detectionImg
        self.detectionValue = detectionValue
        self.centerPoint = centerPoint

def DisplayImage(imgPath):
    img = plt.imread(imgPath)

    plt.imshow(img)
    plt.show()

def DisplayImageSplit(imgPath, fromRow, toRow):
    img = plt.imread(imgPath)

    plt.imshow(img[fromRow:toRow])
    plt.show()

def DisplayImageWithMask(imgPath):
    img = plt.imread(imgPath)
    mask = SelectColorRange(img, config.RGB_Lower_Bound, config.RGB_Upper_Bound)
    plt.imshow(mask)
    plt.show()

def GetScaledTemplates(templatePath,
                     additional_template_scales,
                     run_at_half_scale):
    """
    Returns an array of TemplateImageData that has been scaled
    from 1x size to the additional sizes. If running at half size
    will take the resulting templates and scale them in half.

    :param param1: templatePath -- String path to template image
    :param param2: additional_template_scales -- Array of template scales ie [.8, 1.1]
    :param param3: run_at_half_scale -- Boolean if running template detection at half the capture scale
    :returns: Array[TemplateImageData]
    """
    scaledTemplates = [None] * (len(additional_template_scales) + 1)
    
    scaledTemplates[0] = TemplateImageData()
    scaledTemplates[0].ScaledImage = cv.imread(templatePath, -1)
    scaledTemplates[0].ScaleValue = 1
    scaledTemplates[0].Width, scaledTemplates[0].Height = scaledTemplates[0].ScaledImage.shape[::-1]
    
    scaledTemplates = ScaleTemplates(scaledTemplates, additional_template_scales)
    
    if run_at_half_scale:
        for ctr in range(0, len(scaledTemplates)):
            scaledTemplates[ctr].ScaledImage = cv.pyrDown(scaledTemplates[ctr].ScaledImage)
            scaledTemplates[ctr].Width = int(scaledTemplates[ctr].Width / 2)
            scaledTemplates[ctr].Height = int(scaledTemplates[ctr].Height / 2)

    return scaledTemplates

def GetScreenShot(monitor_number = 1):
    with mss.mss() as sct:
        mon = sct.monitors[monitor_number]
        img = np.array(sct.grab(mon))
        rgbImg = cv.cvtColor(img, cv.COLOR_BGRA2RGB)
        return rgbImg

def StreamScreen(monitor_number = 1, delay=20):
    while True:
        with mss.mss() as sct:
            mon = sct.monitors[monitor_number]
            img = np.array(sct.grab(mon))
            cv.imshow("OpenCV", img)
            cv.waitKey(delay)

def RunTemplateDetection(img,
                        scaledTemplates,
                        run_at_half_scale,
                        rgb_lower_bound,
                        rgb_upper_bound,
                        display_original_image):
    """
    Prepares the image masks and runs the template detection at the various
    scales. Can be run on the image as is or can split the image in half and
    run template detection against the top and bottom half of the image 
    multithreaded for some performance gain. The image can also be scale to
    half its size for a performance gain.

    :param param1: img -- Current image to run detection on.
    
    :param param2: scaledTemplated -- Array[TemplateImageData] of single 
    template at multiple scales
    
    :param param3: run_at_half_scale -- Boolean if running template detection 
    at half the image scale
    
    :param param3: rgb_lower_bound -- Numpy RGB Array of lower color bound to
    generate the mask ie [0,0,0]
    
    :param param4: rgb_upper_bound -- Numpy RGB Array of upper color bound to
    generate the mask ie [255,255,255]
    
    :param param5: display_original_image -- Boolean to return the masked
    image with detection square or original image with detection square

    :returns: DetectionResult
    """

    start_time = time.time()
    
    if run_at_half_scale:
        img = cv.pyrDown(img)
    mask = SelectColorRange(img, rgb_lower_bound, rgb_upper_bound)
    
    detectResult = DetectionResult()

    (detectResult.detectionImg, detectResult.centerPoint, detectResult.detectionValue) = TemplateDetection(scaledTemplates, mask, True, img, display_original_image)
    
    #To correct the colors from a BGR to RGB
    detectResult.detectionImg = cv.cvtColor(detectResult.detectionImg, cv.COLOR_RGB2BGR)
    print("----- %s seconds -----" % (time.time() - start_time))
    print("----- %s fps -----" % (1.0 / (time.time() - start_time)))
    return detectResult

def RunDetectionTest(test_images_path,
                    scaledTemplates,
                    run_at_half_scale,
                    rgb_lower_bound,
                    rgb_upper_bound,
                    display_original_image,
                    ms_wait_between_detections):
    
    testImages = glob.glob(test_images_path)

    
    
    testImgArray = []
    for imgPath in testImages:
        testImgArray.append(plt.imread(imgPath))
    ctr = 0
    while True:
        
        testImg = testImgArray[ctr]
        results = RunTemplateDetection(testImg, scaledTemplates, run_at_half_scale, rgb_lower_bound, rgb_upper_bound, display_original_image)
        cv.imshow("SC Reticle Detection", results.detectionImg)
        cv.waitKey(ms_wait_between_detections)
        print(ctr)
        ctr = (ctr + 1) % 7

def ScaleTemplates(scaledTemplates, templateScales):
    for scale in range(1, len(templateScales)+1):
        scaledTemplates[scale] = TemplateImageData()
        scaledTemplates[scale].ScaledImage = cv.resize(scaledTemplates[0].ScaledImage, 
            dsize=(int(scaledTemplates[0].ScaledImage.shape[1] * templateScales[scale-1]), 
            int(scaledTemplates[0].ScaledImage.shape[0] * templateScales[scale-1])))
        scaledTemplates[scale].ScaleValue = templateScales[scale-1]
        scaledTemplates[scale].Width, scaledTemplates[scale].Height = scaledTemplates[scale].ScaledImage.shape[::-1]
    return scaledTemplates
          
def SelectColorRange(img, lowColor, highColor):
    # img_hsv1 = cv.cvtColor(ducatiImg, cv.COLOR_RGB2BGR)
    # img_hsv = cv.cvtColor(ducatiImgCv, cv.COLOR_BGR2HSV)
    # hsv_color1 = np.asarray([100, 150, 80])
    # hsv_color2 = np.asarray([178, 184, 208])
    
    # mask = cv.inRange(img_hsv, hsv_color1, hsv_color2)
    # plt.imshow(mask, cmap='gray')

    rgbMask = cv.inRange(img, lowColor, highColor)
    # plt.imshow(rgbMask, cmap='gray')
    
    # plt.show()

    # grayMask = cv.cvtColor(rgbMask, cv.COLOR_RGB2GRAY)
    
    
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(rgbMask)
    # plt.show()
    # TemplateDetection(imgTemplate, rgbMask)
    return rgbMask
        
def TemplateDetection(scaledTemplates, img, splitImage=True, origImg=None, returnOrigImg=False):
    
    # All the 6 methods for comparison in a list
    # methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
    #             'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    
    meth = 'cv.TM_CCOEFF_NORMED'
    max_match = 0
    
    method = eval(meth)
    
    min_val, max_val, min_loc, max_loc = (0,0,0,0)
    matched_scale = 0
    w, h = (0,0)
    halfHeight = int(img.shape[0] / 2)
    templ_buffer_height = scaledTemplates[len(scaledTemplates) - 1].Height
    
    if splitImage:
        topImg = img[:halfHeight + templ_buffer_height]
        btmImg = img[(halfHeight-templ_buffer_height):]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 
        future_templ_matches = []
        if splitImage:
            for scaled_template in scaledTemplates:
                future_templ_matches.append(executor.submit(RunTemplateMatch, 
                                            btmImg, 
                                            scaled_template, 
                                            method,
                                            (0, halfHeight - templ_buffer_height)))
            
            for scaled_template in scaledTemplates:
                future_templ_matches.append(executor.submit(RunTemplateMatch, 
                                            topImg, 
                                            scaled_template, 
                                            method,
                                            (0,0)))
        else:
            future_templ_matches = {executor.submit(RunTemplateMatch, img, scaled_template, method): scaled_template for scaled_template in scaledTemplates}

        for templ_match in concurrent.futures.as_completed(future_templ_matches):
            result = templ_match.result()
            if result.max_val >= max_val:
                max_val = result.max_val
                min_val = result.min_val
                max_loc = [sum(x) for x in zip(result.max_loc, result.offSet)]
                min_loc = [sum(x) for x in zip(result.min_loc, result.offSet)]
                matched_scale = result.scale
                w = result.width
                h = result.height
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    print("Matched Scale: ", matched_scale, "Matched Value: ", max_val)
    
    center_point = [ int((top_left[0] + bottom_right[0]) / 2) , int((top_left[1] + bottom_right[1]) / 2) ]

    if returnOrigImg:
        cv.rectangle(origImg,top_left, bottom_right, 255, 2)
        cv.circle(img,center_point,4, [255,255,255])
        return (origImg, center_point, max_val)
    else:
        cv.rectangle(img,top_left, bottom_right, 255, 2)
        cv.circle(img,center_point,4, [255,255,255])
        return (img, center_point, max_val)

def RunTemplateMatch(img, template:TemplateImageData, method, offSet = (0,0)):
    # res = cv.matchTemplate(cv.GaussianBlur(img,(5,5),0),template.ScaledImage, method)
    # res = cv.matchTemplate(img,cv.GaussianBlur(template.ScaledImage, (5,5),0), method)
    res = cv.matchTemplate(cv.GaussianBlur(img,(5,5),0),cv.GaussianBlur(template.ScaledImage, (3,3),0), method)
    # res = cv.matchTemplate(img,template.ScaledImage, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    return MatchInfo(min_val, max_val, min_loc, max_loc, template.ScaleValue, template.Width, template.Height, offSet)
        

if __name__ == "__main__":


    # DisplayImageWithMask("TestImages/sc4.jpg")
    # DisplayImage("TestImages/sc4.jpg")
    # img = plt.imread(path)
    # SelectColorRange(img, config.RGB_Lower_Bound, config.RGB_Upper_Bound)
    # plt.imshow(GetScreenShot())
    
    # plt.show()
    templateList = GetScaledTemplates(config.Template_Path,
                       config.Additional_Template_Scales,
                       config.Run_at_Half_Scale)
    
    # RunDetectionStream(config.Template_Path,
    #                     config.Additional_Template_Scales,
    #                     config.Run_at_Half_Scale,
    #                     config.RGB_Lower_Bound,
    #                     config.RGB_Upper_Bound,
    #                     config.Display_Original_Image,
    #                     config.MS_Wait_Between_Detection)
    
    RunDetectionTest(config.Test_Images_Path,
                        templateList,
                        config.Run_at_Half_Scale,
                        config.RGB_Lower_Bound,
                        config.RGB_Upper_Bound,
                        config.Display_Original_Image,
                        config.MS_Wait_Between_Detection)

'''
Roughly 10fps at Half Size 6/7 Correct
Roughly 2fps at Full Size 7/7 Correct

Template was pulled from an original image that is
3584x2240. Sc4.png. Not sure why the screen capture is larger
than the screen. ratio of 1.6

MBP Screen 3072x1920 ratio 1.6

Common Resolution
1920x1080 ratio 1.78 multiple of template .535
3072x1920 ratio 1.6 multiple of template .85 (Still matches when full screened on RunDetectionStream)
3840x2160 ratio 1.85 multiple of template 1.07
3440x1440 ratio 2.39 multiple of template .960

'''