import numpy as np

# Template
# RGB_Lower_Bound = np.asarray([0,0,0])
# RGB_Upper_Bound = np.asarray([255,255,255])

# Find Red Target Box
RGB_Lower_Bound = np.asarray([130, 110, 110])
RGB_Upper_Bound = np.asarray([250, 180, 180])

# Find Targeting Reticle
# RGB_Lower_Bound = np.asarray([170, 200, 200])
# RGB_Upper_Bound = np.asarray([245, 254, 254])

Template_Path = "./Template.jpg"
# Template_Path = "./ReticleTemplate.jpg"
Run_at_Half_Scale = True
Additional_Template_Scales = [.6, .8, 1.1]
Display_Original_Image = True
MS_Wait_Between_Detection = 3000

Test_Images_Path = "./TestImages/*.jpg"