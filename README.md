Advanced Lane Detection
===
In this Advanced Lane Detection project, we apply computer vision techniques to augment video output with a detected road lane, road radius curvature and road centre offset. The video was supplied by Udacity and captured using the middle camera.

![sample lane detection result](https://raw.githubusercontent.com/hortovanyi/udacity-advanced-lane-finding-project/master/output_images/writeup_intro_road.gif)

The goals / steps of this project are the following:

- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

A jupyter/iPython data science notebook was used and can be found on github [Full Project Repo](https://github.com/hortovanyi/udacity-advanced-lane-finding-project) - [Advanced Lane Finding Project Notebook](https://github.com/hortovanyi/udacity-advanced-lane-finding-project/blob/master/Advanced%20Lane%20Finding%20Project.ipynb) (Note the interactive  ipywidgets are not functional on github). The project is written in python and utilises [numpy](http://www.numpy.org/) and [OpenCV](http://opencv.org/).

##Camera Calibration
Every camera has some distortion factor in its lens. The known approach to correct for that in (x,y,z) space is apply coefficients to undistort the image. To calculate this a camera calibration process is required.

It involves reading a set of [warped chessboard images](https://github.com/hortovanyi/udacity-advanced-lane-finding-project/tree/master/camera_cal), converting them into grey scale images before using `cv2.findChessboardCorners()` to identify the corners as `imgpoints`.
![9x6 Chessboard Corners Detected](https://github.com/hortovanyi/udacity-advanced-lane-finding-project/blob/master/output_images/chessboard-corners-detected.png?raw=true)   

If corners are detected then they are collected as image points `imgpoints` along with a set of object points `objpoints`; with an assumption made that the chessboard is fixed on the (x,y) plane at z=0 (object points will hence be the same for each calibration image).

In the function `camera_calibrate` I pass the collected `objpoints`, `imgpoints` and a test image for the camera image dimensions. It in turn uses `cv2.calibrateCamera()` to calculate the distortion coefficients before the test image is undistorted with `cv2.undistort()` giving the following result.
![Original and Undistorted image](https://github.com/hortovanyi/udacity-advanced-lane-finding-project/blob/master/output_images/camera_original_undistort.png?raw=true)

## Pipeline (Test images)

After camera calibration a set of functions have been created to work on test images before later being used in a video pipeline.

### Distortion corrected image
The `undistort_image` takes an image and defaults the `mtx` and `dist` variables from the previous camera calibration before returning the undistorted image.
![test image distorted and undistorted](https://github.com/hortovanyi/udacity-advanced-lane-finding-project/blob/master/output_images/test_image_and_undistorted.png?raw=true)

### Threshold binary images
A threshold binary image, as the name infers, contains a representation of the original image but in binary `0`,`1` as opposed to a BGR (Blue, Green, Red) colour spectrum. The threshold part means that say the Red colour channel( with a range of 0-255) was between a threshold value range of 170-255, that it would be set to `1`.

A sample output follows.
![Sample Threshold Image](https://github.com/hortovanyi/udacity-advanced-lane-finding-project/blob/master/output_images/daytime-normal-threshold.png?raw=true)

Initial experimentation occurred in a [separate notebook](https://github.com/hortovanyi/udacity-advanced-lane-finding-project/blob/master/Thresholds.ipynb) before being refactored back into the project notebook in the `combined_threshold` function. It has a number of default thresholds for sobel gradient x&y, sobel magnitude, sober direction, Saturation (from HLS), Red (from RGB) and Y (luminance from [YUV](https://en.wikipedia.org/wiki/YUV)) plus a `threshold` type parameter (`daytime-normal`, `daytime-bright`, `daytime-shadow`, `daytime-filter-pavement`).

Whilst the `daytime-normal` threshold worked great for the majority of images there were situations where it didn't e.g. pavement colour changes in bright light and shadow.

![Daytime Normal with noise bright light & pavement change](https://github.com/hortovanyi/udacity-advanced-lane-finding-project/blob/master/output_images/daytime-normal-overexposed.png?raw=true)

![Daytime Normal with shadow](https://github.com/hortovanyi/udacity-advanced-lane-finding-project/blob/master/output_images/daytime-normal-shadow.png?raw=true)

Other samples [Daytime Bright](https://github.com/hortovanyi/udacity-advanced-lane-finding-project/blob/master/output_images/daytime-bright.png?raw=true), [Daytime Shadow](https://github.com/hortovanyi/udacity-advanced-lane-finding-project/blob/master/output_images/daytime-shadow.png?raw=true) and [Daytime Filter Pavement](https://github.com/hortovanyi/udacity-advanced-lane-finding-project/blob/master/output_images/daytime-filter-pavement.png?raw=true).

### Perspective transform - birds eye view
To be able to detect the road lines, the undistorted image is warped. The function `calc_warp_points` takes an image's height & width and then calculates the `src` and `dst` array of points. `perspective_transforms` takes them and returns two matrixes `M` and `Minv` for  `perspective_warp` and `perpective_unwarp` functions respectively. The following image, shows an undistorted image, with the src points drawn with the corresponding warped image (the goal here was straight lines) ![Distorted with bird's eye view](https://github.com/hortovanyi/udacity-advanced-lane-finding-project/blob/master/output_images/undistorted_with_birdseye.png?raw=true)

### Lane-line pixel identification and polynomial fit
Once we have a birds eye view with a combined threshold we are in a position to identify lines and a polynomial to draw a line (or to search for points in a binary image).

![topdown warped binary image](https://github.com/hortovanyi/udacity-advanced-lane-finding-project/blob/master/output_images/birdseye-combined-straighline-small.png?raw=true)

A histogram is created via `lane_histogram` from the bottom third of the topdown warped binary image. Within `lane_peaks`, `scipy.signal` is used to identify left and right peaks. If just one peak then the max bin either side of centre is returned.

`calc_lane_windows` uses these peaks along with a binary image to initialise a left and right instance of a `WindowBox` class. `find_lane_window` then controls the WindowBox search up the image to return an array of `WindowBox`es that should contain the lane line. `calc_fit_from_boxes` returns a polynomial or None if nothing found.

`poly_fitx` function takes a `fity` where
```fity = np.linspace(0, height-1, height)``` and a polynomial to calculate an array of x values.

The search result is plotted on the bottom left of the below image with each box in green. To test line searching by polynomial, I then use the left & right WindowBox search polynomials as input to `calc_lr_fit_from_polys`. The bottom right graphic has the new polynomial line draw with a blue search window (relates to polynomial used for the search from `WindBox`es) that was used overlapping with a green window for the new.

![Warped box seek and new polynomial fit](https://github.com/hortovanyi/udacity-advanced-lane-finding-project/blob/master/output_images/warped-box-seek-with-new-polyfit.png?raw=true)

### Radius of curvature calculation and vehicle from centre offset
In road design, curvature is important and its normally measured by its radius length. For a straight line road, that value can be quite high.

In this project our images are in pixel space and need to be converted into meters. The images are of US roads and I measured from [this image](https://github.com/hortovanyi/udacity-advanced-lane-finding-project/blob/master/output_images/birdseye-combined-straighline.png?raw=true) the distance between lines (413 pix) and the height of dashes (275 px). Lane width in the US is ~ 3.7 meters and dashed lines 3 metres. Thus `xm_per_pix` = 3.7/413 and `ym_per_pix` = 3./275 were used in `calc_curvature`. The function converted the polynomial from pixel space into a polynomial in meters.  

To calculate the offset from centre, I first determined where on the x plane, both the left `lx` and right `rx` lines crossed the image near the driver. I then calculated the `xcentre` of the image as the width/2. The `offset` was calculated such `(rx - xcenter) - (xcenter - lx)` before being multiple by `xm_per_pix`.

### Final pipeline
I decided to take a more python class based approach once I progressed through this project. Inside the classes, I called the functions mentioned previously. The classes created were:

- `Lane` contains image processing, final calculations for view drawing and reference to left and right `RoadLine`s. It also handled searching for initial lines, recalculations and reprocessing a line that was not sane;
- `RoadLine` contains a history of `Line`s and associated curvature and plotting calculations using weighted means; and
- `Line` contains detailed about the line and helper functions

Processing is triggered by setting the `Lane.image` variable. Convenient property methods `Lane.warped`, `Lane.warped_decorated`, `lane.result` and `lane.result_decorated` return processed images. It made it very easy to debug output using interactive ipywidgets (which don't work on github)

#### Sample result images
![`lane.result_decorated`](https://github.com/hortovanyi/udacity-advanced-lane-finding-project/blob/master/output_images/result_with_lane_unwarped.jpg?raw=true)

![Lane.warped_decorated](https://github.com/hortovanyi/udacity-advanced-lane-finding-project/blob/master/output_images/birdseye-with-original-threshold-result-using-windowing.jpg?raw=true)

## Pipeline (Video)

Using [moviepy](http://zulko.github.io/moviepy/) to process the project video was simple. I also decorated the result with a frame count. The [Project Video Lane mp4 on GitHub](https://github.com/hortovanyi/udacity-advanced-lane-finding-project/blob/master/project_video_lane.mp4?raw=true), contains the result ([YouTube Copy](https://www.youtube.com/watch?v=cQU2tNtslZ0))

[![Result Video embedded from YouTube](http://img.youtube.com/vi/cQU2tNtslZ0/0.jpg)](https://www.youtube.com/watch?v=cQU2tNtslZ0)

## Discussion
### Problems/Issues faced
To some degree, I got distracted with trying to solve the issues I found in my algorithm with the challenge videos. This highlighted, that I need to improve my understanding of colour spaces, sobel and threshold combinations.

I included a basic algorithm to remove pavement colours from the images using a centre, left and right focal point. I noticed that the dust colour on the vehicle seemed to be also in the road side foliage. This however wasn't sufficient to remove all pavement colour and didn't work when there was a road type transition. It was very CPU intensive.

In the end, I used a combination of different methods, that used a basic noise filter on warped binary images to determine, if it was sufficient to look for a line or not. If it wasn't it tried the next one, with the final being a vertical rectangle window crawl down the image. Where the best filter was determined for each box. Again this was CPU intensive, but worked.

Another issue faced was using the previous curvature radius to determine if this line was sane or not. The values were too jittery and when driving on a straight line, high. I decided not to pursue this.

### Opportunities for improvement in the algorithm/pipeline
There is room here for some refactoring into a more Object oriented approach. This was not evident at the start of the project as to how it should be structured. I experimented a little with using `Pool` from multiprocessing to parallelise left and right lane searches. It didn't make it into my final classes as for normal line searching using a polynomial, as I did not ascertain if the multiprocessing overhead, outweighed the parallelism value. Certainly potential here to use a more functional approach to give the best runtime options for parallelisation.

Other areas, include automatically detecting the src points for warp, handling bounce in the road and understanding surface height (above road) of the camera and its impact.

I thought also as I've kept history, I could extend the warp to include a bird'e eye representation of the car on the road and directly behind it. I did mean averaging on results for smoothing drawn lines, but this was not included in the new line calculations from the next image frames.

The algorithm could also be made to make predictions about the line when there is gaps. This would be easier with continuous lines then dashed.

### Hypothetical pipeline failure cases
Pavement fixes and/or combined with other surfaces that create vertical lines near existing road lines.

It would also fail if there was a road crossing or a need to cross lanes or to exit the freeway.

Rain and snow would also have an impact and I'm not sure about night time.

Tail gating a car or a car on a tighter curve would potentially interrupt the visible camera and hence line detection.
