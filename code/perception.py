import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh_min=(160, 160, 160), rgb_thresh_max = (255,255,255)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    within_thresh = (img[:,:,0] > rgb_thresh_min[0]) \
            & (img[:,:,1] > rgb_thresh_min[1]) \
            & (img[:,:,2] > rgb_thresh_min[2]) \
            & (img[:,:,0] < rgb_thresh_max[0]) \
            & (img[:,:,1] < rgb_thresh_max[1]) \
            & (img[:,:,2] < rgb_thresh_max[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[within_thresh] = 1
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the
    # center bottom of the image.
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle)
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))

    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    mask = cv2.warpPerspective(np.ones_like(img[:,:,0]), M, (img.shape[1], img.shape[0]))
    return warped, mask


def find_rocks(img, levels=(110,110, 50)):
    rockpix = ((img[:,:,0] > levels[0]) \
                & (img[:,:,1] > levels[1]) \
                & (img[:,:,2] < levels[2]))

    color_select = np.zeros_like(img[:,:,0])
    color_select[rockpix] = 1

    return color_select

def remap_values(value, inMin, inMax, outMin, outMax):
    # Figure out how 'wide' each range is
    inSpan = inMax - inMin
    outSpan = outMax - outMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - inMin) / float(inSpan)

    # Convert the 0-1 range into a value in the right range.
    return outMin + (valueScaled * outSpan)

# Define a function to mask the navigable terrain pixels
def mask_navigable(nav_binary):

    H_start_percent = 0 # percent value
    H_end_percent = 60 # percent value
    V_start_percent = 0 # percent value
    V_end_percent = 85 # percent value

    driving_mask = np.zeros((nav_binary.shape[0], nav_binary.shape[1])) # initialize matrix of zeros

    H_start_col = int(round(remap_values(H_start_percent, 0, 100, 0, 0)))
    H_end_col = int(round(remap_values(H_end_percent, 0, 100, 0, nav_binary.shape[1])))
    V_start_col = int(round(remap_values(V_start_percent, 0, 100, 0, 0)))
    V_end_col = int(round(remap_values(V_end_percent, 0, 100, 0, nav_binary.shape[0])))
    driving_mask[V_start_col:V_end_col,H_start_col:H_end_col] = 1 # select range of rows and columns

    mask_nav = nav_binary * driving_mask

    return mask_nav

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO:
    # NOTE: camera image is coming to you in Rover.img
    dst_size = 5
    # 1) Define source and destination points for perspective transform
    source = np.float32([[200, 95],
                 [300, 140],
                 [10, 140],[118, 95]])
    destination = np.float32([[165, 135],
                 [165, 145],
                 [155, 145],
                 [155, 135]])
    # 2) Apply perspective transform
    warped, mask = perspect_transform(Rover.img, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    rgb_nav_min = (160,160,160)
    rgb_nav_max = (255, 255, 255)
    navigable_threshed = color_thresh(warped, rgb_nav_min, rgb_nav_max)

    rgb_obs_min = (0,0,0)
    rgb_obs_max = (170,170,170)
    threshed_obstacles = color_thresh(warped,rgb_obs_min, rgb_obs_max)

    rgb_rock_min = (110, 110, 5)
    rgb_rock_max = (210, 210, 145)
    threshed_rocks = color_thresh(warped, rgb_rock_min, rgb_rock_max)

    obs_map = np.float32(threshed_obstacles) * mask

    navigable_masked = mask_navigable(navigable_threshed)

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    Rover.vision_image[:,:,2] = navigable_masked * 255
    Rover.vision_image[:,:,0] = obs_map * 255
    # 5) Convert map image pixel values to rover-centric coords
    x_nav_pix, y_nav_pix = rover_coords(navigable_threshed)
    x_obs_pix, y_obs_pix = rover_coords(threshed_obstacles)
    x_navM_pix, y_navM_pix = rover_coords(navigable_masked)
    # 6) Convert rover-centric pixel values to world coordinates
    world_size = Rover.worldmap.shape[0]
    scale = 2 * dst_size
    navigable_x_world, navigable_y_world = pix_to_world(x_nav_pix, y_nav_pix, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
    obstacle_x_world, obstacle_y_world = pix_to_world(x_obs_pix, y_obs_pix, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale )
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    # check that the rover is on even ground
    if ((Rover.pitch <= 1) or (Rover.pitch >= 359)) and ((Rover.roll <= 1) or (Rover.roll >= 359)):
        # to account for overlap between the two
        Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1   # worldmap in the blue channel (2) we will update to be 255 where we find nav terrain
        Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1 # worldmap in the red channel (0) will be 255 where we find obstacles

        #obs_pix = data.worldmap[:,:, 0] > 1 # if the red channel is > 0 (navigable terrain) I will just assume it is and...
        #data.worldmap[obs_pix,2] = 0 # ...set the blue channel to zero (discard the obstacles)
        #data.worldmap[obs_pix,0] = 255 # ...set the red channel to 255

        nav_pix = Rover.worldmap[:,:, 2] > 2 # if the blue channel is > 0 (navigable terrain) I will just assume it is and...
        Rover.worldmap[nav_pix,0] = 0 # ...set the red channel to zero (discard the obstacles)
        Rover.worldmap[nav_pix,2] = 255 # ...set the blue channel to 255

        obs_pix = Rover.worldmap[:,:, 0] > 6 # if the red channel is > 0 (navigable terrain) I will just assume it is and...
        #data.worldmap[obs_pix,2] = 0 # ...set the blue channel to zero (discard the obstacles)
        Rover.worldmap[obs_pix,0] = 255 # ...set the red channel to 255
    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    dist_nav, angles_nav = to_polar_coords(x_nav_pix, y_nav_pix)
    dist_obs, angles_obs = to_polar_coords(x_obs_pix, y_obs_pix)
    dist_navM, angles_navM = to_polar_coords(x_navM_pix, y_navM_pix)
    Rover.nav_angles = angles_navM
    Rover.obs_angles = angles_obs

    #See if we can find some rocks
    rock_map = find_rocks(warped, levels=(110,110,50))
    rock_x, rock_y = rover_coords(rock_map)
    Rover.rock_dist, Rover.rock_ang = to_polar_coords(rock_x, rock_y)
    if np.count_nonzero(Rover.rock_ang) < 3:
        Rover.rock_ang = None
    if rock_map.any(): # gives True if at least 1 element of rock_map is True, otherwise False

        rock_x_world, rock_y_world = pix_to_world(rock_x, rock_y, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
        rock_idx = np.argmin(Rover.rock_dist) # minimum distance rock pixel
        rock_xcen = rock_x_world[rock_idx]
        rock_ycen = rock_y_world[rock_idx]
        if Rover.rock_ang is not None:
            Rover.steer_cache = np.clip(np.mean(Rover.rock_ang * 180/np.pi), -15, 15)

        Rover.worldmap[rock_ycen,rock_xcen, 1] = 255 # update the rover world map to be 255 at that center point
        Rover.vision_image[:,:,1] = rock_map * 255 # put those rock pixels onto the vision image
    else:
        Rover.vision_image[:,:,1] = 0

    return Rover
