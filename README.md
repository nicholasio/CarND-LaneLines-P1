I recently joined the Udacity [Self Driving Car Nanodegree](https://www.google.com.br/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwj4krnB4p3YAhUFjZAKHcKYBWIQFggwMAA&url=https%3A%2F%2Fwww.udacity.com%2Fcourse%2Fself-driving-car-engineer-nanodegree--nd013&usg=AOvVaw2uaM9n2beuYSiTnL1Sh6sT). The nanodegree is amazing and I'm really enjoying taking it. The nanodegree is comprised of several projects and the first project (and the simplest one) is to build a pipeline to identify lanes on a road. Here I'm gonna present my solution to this project. The code is written in Python and I'm using the [OpenCV](https://opencv.org/) library. Identifying lane lines is a common and simple task that self driving cars need to perform. There are a bunch of techniques out there. Since this is the first project, we're suppose to build a more simpler approach. Our goal here is to identify straight lines in order to "fit" left and right lane markings. This approach works well when a car is driving straight but it starts to perform badly on curves (and you'll see an example of this at the end of this article).

### Overview of the pipeline

Before digging into each step taken, let me summarize what I'm doing in order to identify the lanes markings.

1.  Convert Images to HSL
2.  Apply a white and yellow color masks to the image
3.  Apply a Gaussian filter
4.  Run Canny Edges detection to identify the edges
5.  Get the image ROI (region of interest) to focus on the road
6.  Run Hough Transform to find lines in the image
7.  Separate lines into left and right lines
8.  Find the best fit line for the left and right line points
9.  Extrapolate the lines in order to get one single line that cover the entire lane line

### Pre-processing the image

The first three steps does some pre-processing to the image that will make our life easier in the next steps. First I convert the image color space from RGB to HSL, this gives us slightly better results for color thresholding compared to RGB. HSL is just another model for representing colors. There are several image color models such as RGB, YUV, YCrCb, CMYK, I'm not gonna dive into details of color representation but you can think of them as different ways to represent colors in images. It's also important to note that there are equations that can convert colors from one representation to another. If you're interest in reading more check out this Wikipedia [article](https://en.wikipedia.org/wiki/Color_model). in OpenCV converting color models is as simple as calling a function: \[code language="python"\] img = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) \[/code\] Next we apply a technique called color thresholding to select pixels whose color intensity values are between the given color range. For example, if we want to select only white regions of the image, we could look for pixels with RGB values of (255,255,255) since it represents white. However, there are various levels of white and if we only look for the "full" white we would be missing tons of pixels in the image. A better approach is to get pixels withing a given range like (200,200,200) to (255,255,255). The sequence of images below shows the result of applying two color masks to get yellow and white lane markings. The last image is the two masked combined. ![](https://blognicholasandre.files.wordpress.com/2017/12/color_maks.png?w=700) I then use the combined mask to get only the pixels of interest in the original image. The image on the left is the result after applying the Gaussian filter which basically blurs the image. The goal of applying the Gaussian filter is to remove noise and avoid noisy edges to be detected. Compare the images below, you should be able to see that noisy pixels around the lane markings have been smoothed out. ![](https://blognicholasandre.files.wordpress.com/2017/12/pixels_of_interest_and_gaussian.png?w=700)

### Finding the edges

Now that I have filtered out most of the pixels of the image that I'm not interest in, I run the canny edge detection algorithm. Basically it uses the gradient of the image with a low and high threshold parameters to find edges. If you're interest in learning more about Canny Edge check out this [link](http://aishack.in/tutorials/canny-edge-detector/). I used 50 and 150 for the low and high threshold respectively. ![](https://blognicholasandre.files.wordpress.com/2017/12/canny.png) The canny edge algorithm returns a binary image with white pixels belonging to the edges the algorithm detected. There are still some unwanted edges found in the image, but since the camera is always in a fixed position we can simply define a region of interest and discard everything outside this region. ![](https://blognicholasandre.files.wordpress.com/2017/12/roi.png?w=700) Cool! with a region of interest I got only pixels belonging to the lane markings! The next steps is to find actual line points from this image.

### Hough Transform and extrapolating lines

This is probably the most complicated step. Here I first apply the [Hough Transform](https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html) Algorithm. Basically, it receives the output of any edge detection algorithms such as canny edges and tries to find lines. The algorithm is a bit complex but the way it works is beautiful! Check out this [article](https://alyssaq.github.io/2014/understanding-hough-transform/) if you're interest in learning how to implement it. For applying the Hough Transform I decided to use following parameters:

*   Rho: 1
*   Theta: PI/180
*   Threshold: 30
*   Min Line Length: 10
*   Max Line Length: 150

Initially I was using lower values for the Max Line Length, but increasing it to a higher value improved the line detection as it was able to "connect" the dashed white lane markings. Bellow is the output of the raw lines detected plotted onto the original image. ![](https://blognicholasandre.files.wordpress.com/2017/12/raw_lines_img.png?w=300) As you can see it detects multiples lines for each lane markings. So we need a way to average these lines. The idea is to find two lines, one for the left lane and another for the right lane. So the first step is to separate the lines found. The way we can do that is by computing the slope of the line:

*   If slope is positive (as y decreases, x decreases as well), the line belongs to the right lane.
*   If the slope is negative (as y decrease, x increases), the line belongs to the left lane.

Once the lines are properly grouped, I use line the points to find a best fit line using the least square method. In python it's possible to easily calculate that by using [np.polyfit()](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.polyfit.html). This function will return the coeficientes of the line equation (y = ax + b). With this equation I can easily plot a line from the bottom the of region of interest to the top of it. The image below shows the final output of the pipeline, previous steps are also shown for better understanding. ![](https://blognicholasandre.files.wordpress.com/2017/12/test1.png?w=700)

### Tests on videos

Video 1 \[embed\]https://cldup.com/4BVokJbvKf.mp4\[/embed\] Video 2 \[embed\]https://cldup.com/OTj4\_SA0sR.mp4\[/embed\] Video 3 This video is more challenging because it more curves. Since we're only using straight lines the solution does not perform very well, but it is still able to detect the lane lines. \[embed\]https://cldup.com/\_D0bXsmr1Q.mp4\[/embed\]

Areas to improve
----------------

*   Straight Lines aren't able to detect lane lines properly in curves. One potential solution is to use quadratic functions to fit the lane markings.
*   The current pipeline always assume that lane markings are either yellow or white. By parameter tuning it should be possible to completely get rid of color masks and use only edge detection.
*   Use some sort of memory to average the current line with previous lines in order to make the detected lines more stable.