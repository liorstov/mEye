# mEye

## Dehomogenize
Converting P coordinates to 3D. Convert to camera image using the focal length (800) and then to Ego ant to Ego 0
## Homogenize
Assuming  dy = 0 . Use Y coord from real world and get Z and X. Convert to ego

    """get Y coordinate from world first frame assuming dy=0"""  
	Y = convertWorldToCamera(frame,pixel)[1,0]  	  
	Z = Y*focallength/V  
	X = U*Z/focallength
	
	"""convert to ego"""  
	cameraCords =  np.hstack([[X,Y,Z],1])  
	WorldCS = np.linalg.inv(T_c_e[frame].reshape(4,4))[:3]*cameraCords.reshape(4,1)
	
## Calculate heading change
get heading change between frame and previous frame. 

 1. calculated difference between all frames 
 2. Get car length from real world (assuming furthest points)
 3. If all pixels moved the same distance return 0
 4. if not the calculate according to bicycle model
	 1. Assuming the rear has the lowest dz and fron has the biggest 
	 2. steer direction is the angle between frame surface and vector between current and previous front pixel

	    `steer = getConvertedFramePixel(frame, front,converted) - getConvertedFramePixel(frame - 1, front,converted)`
	 3. velocity vector = size of change of rear pixel
						
			"""calculate heading according to bicycle model"""  
			delta = vg.signed_angle(surface,steer,np.array([0,1,0]), units="rad")
			R = carLength/math.tan(delta)  
			HeadingChange = velo/R

##  TransformTwoCentroids(pointsCloud):
This function is an attempt to correct noisy data.
calculate the median of Z coordinates and divide the pixel cloud to two. 
Calculate the centroids for the two clouds.
create a surface from two centroids

    median = np.median(group[:, 1])  
	cent1 = np.mean(pointsCloud.loc[(pointsCloud['Frame'] == frame) & (pointsCloud["Z"] > median)]).to_numpy()[2:5]  
	cent2 = np.mean(pointsCloud.loc[(pointsCloud['Frame'] == frame) & (pointsCloud["Z"] <= median)]).to_numpy()[2:5]

 ## lineraRegressionFrame(frame,converted)
 This function estimates the dispersion of a frame using R<sup>2</sup> score. Trearing the pixel cloud on (X,Z) plane.
 

# calculatYawFromWorld(frame):
   calculate the yaw angle from world frame

## mainFunction():
1. calculate the first heading from the first world frame (given
2. calculate car length from first world frame (given)
3. convert camera frames to Ego 0
4. calculate two  centroids for each frame for later use
5. calculate regression on frames to estimate noise
6. calculate noise bar and skip bar according to percentiles of R2 score. noiseBar  is 85th percentile and skip is 5th.  
	1. if frame noise is higher then noise bar the use centroids instead
	2. If frame noise is higher then skip bar then it is skipped
7. start iterating on frames
	1. Check frame R2 and decide if to skip it or to use centroids or do continue with given points
	2. calculate heading using heading function
	3. calculate error between world heeding and calculated heading
	4. add heading change to current heading
	5. calculate target's centroid dx dy dz
	6. if frame number %5 ==0 ground heading

## results

# Target only

![Alt Text](/plots/targetOnly.png)
### Mean error on heading : 0.0
# Target and Ego
![Alt Text](/plots/TargetEgo.png)
### Mean error on heading : 0.0
# target only noise
![Alt Text](/plots/noisetargetOnly.png)
### Mean error on heading : -0.143914985

# target and Ego noise
![Alt Text](/plots/TargetEgoNoise.png)
### Mean Error on heading : 0.014454757

## attached files
the csv files contain a list describing each frame: 
heading: calculated heading
real heading: calculated from worlds
X Y Z the target position in ego0
dx dy dz: difference ffrom las frame
dth: change in heading
Rsqr: R2 score for the pixel cloud
error: error on the calculated heading
## comments
I estimated the error base on pixel cloud dispersion and then decided weather to ignore te change or correct with centroids or do nothing.
I estimated these bars according to Rsqr distribution in the frame set using percentiles.
I chose the percentiles by try and error but it is possible to train the model or to find some optimal point.
I now think it is better to skip then use centroids since i can corrrect myself every 5 frames.

The handling error estmation was done posterior to the model run, i didnt use it to evaluate !!!
