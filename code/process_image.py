def process_image(img):
    # 1) Define source and destination points for perspective transform
    dst_size = 5
    source = np.float32([[200, 95],[300, 140],[10, 140],[118, 95]])
    destination = np.float32([[165, 135],[165, 145],[155, 145],[155, 135]])
    # 2) Apply perspective transform
    warped = perspect_transform(img, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    rgb_nav_min = (170,170,170)
    rgb_nav_max = (255, 255, 255)
    navigable_threshed = color_thresh(warped, rgb_nav_min, rgb_nav_max)
    rgb_obs_min = (0,0,0)
    rgb_obs_max = (170,170,170)
    threshed_obstacles = color_thresh(warped,rgb_obs_min, rgb_obs_max)
    rgb_rock_min = (110, 110, 5)
    rgb_rock_max = (210, 210, 145)
    threshed_rocks = color_thresh(warped, rgb_rock_min, rgb_rock_max)
    # 4) Convert thresholded image pixel values to rover-centric coords
    x_nav_px, y_nav_px = rover_coords(navigable_threshed)
    x_obs_px, y_obs_px = rover_coords(threshed_obstacles)
    # 5) Convert rover-centric pixel values to world coords
    xpos = data.xpos[data.count]
    ypos = data.ypos[data.count]
    yaw = data.yaw[data.count]
    w_size = data.worldmap.shape[0]
    scale = 2 * dst_size # defines how big squares are in the perspective transform
    nav_x_w, nav_y_w = pix_to_world(x_nav_px, y_nav_px, xpos, ypos, yaw, w_size, scale)
    obs_x_w, obs_y_w = pix_to_world(x_obs_px, y_obs_px, xpos, ypos, yaw, w_size, scale)
    # to account for overlap between the two
    data.worldmap[nav_y_w, nav_x_w, 2] += 1
    data.worldmap[obs_y_w, obs_x_w, 0] += 1
    nav_pix = data.worldmap[:,:, 2] > 2 # blue channel
    data.worldmap[nav_pix,0] = 0 # red channel
    data.worldmap[nav_pix,2] = 255 # blue channel
    obs_pix = data.worldmap[:,:, 0] > 6 # red channel
    data.worldmap[obs_pix,0] = 255 # red channel

    output_image = np.zeros((img.shape[0] + data.worldmap.shape[0], img.shape[1]*2, 3))
    # Populate regions of the image with various output
    output_image[0:img.shape[0], 0:img.shape[1]] = img
    warped = perspect_transform(img, source, destination)
    output_image[0:img.shape[0], img.shape[1]:] = warped # warped image in the upper right
    # Overlay worldmap with ground truth map
    map_add = cv2.addWeighted(data.worldmap, 1, data.ground_truth, 0.5, 0)
    # Flip map overlay so y-axis points upward and add to output_image
    output_image[img.shape[0]:, 0:data.worldmap.shape[1]] = np.flipud(map_add)
    # Put some text over the image
    cv2.putText(output_image,"Populate this image with your analyses to make a video!", (20, 20),
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    data.count += 1 # Keep track of the index in the Databucket()
    return output_image
