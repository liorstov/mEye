from idlelib.format import Rstrip
from lib2to3.main import diff_texts

import pandas as pn
import numpy as np
import matplotlib.pyplot as plt
import vg
import math
from scipy import stats
from scipy.spatial.distance import pdist , squareform



K = np.matrix([[800,0,0],[0,800,0],[0,0,1]])
maxFrames = 50
numPixels = 60
focallength = 800
dt = 1
world = pn.read_csv("C:\\Users\\Lior\\SandBox\\untitled\\only_target\\p.csv")
p = pn.read_csv("C:\\Users\\Lior\\SandBox\\untitled\\only_target\\pp.csv")
Conversion = pn.read_csv("C:\\Users\\Lior\\SandBox\\untitled\\only_target\\conversion_matrices.csv")
T_c_e = Conversion['T_c_e'].apply(np.matrix)
T_e_e0 = Conversion['T_e_e0'].apply(np.matrix)
T_t_e = Conversion['T_t_e'].apply(np.matrix)
T_t_t0 = Conversion['T_t_t0'].apply(np.matrix)



def estimateDistance():
    """estimate the distance of the object from the first frame"""
    worldPixel = world.loc[(world['Frame']==0) & (world['Pixel']==0), ['X', 'Y', 'Z', 'home']].to_numpy().reshape(4, 1)
    cameraImage = (T_c_e[0].reshape(4, 4) * T_e_e0[0].reshape(4, 4))[:3] * worldPixel
    return cameraImage.item(2)

def convertWorldToEgo(frame,pixel):
    worldPixel = world.loc[(world['Frame']==frame) & (world['Pixel']==pixel), ['X', 'Y', 'Z', 'home']].to_numpy().reshape(4, 1)
    return (T_e_e0[frame].reshape(4, 4))[:3]*worldPixel

def  convertWorldToEgoToTarget(frame,pixel):
    worldPixel = world.loc[(world['Frame']==frame) & (world['Pixel']==pixel), ['X', 'Y', 'Z', 'home']].to_numpy().reshape(4, 1)
    return (T_t_e[frame].reshape(4,4)*T_e_e0[frame].reshape(4, 4))[:3]*worldPixel

def convertWorldToEgoToTargetToT0(frame,pixel):
    worldPixel = world.loc[(world['Frame']==frame) & (world['Pixel']==pixel), ['X', 'Y', 'Z', 'home']].to_numpy().reshape(4, 1)
    return (T_t_t0[frame].reshape(4,4)*T_t_e[frame].reshape(4,4)*T_e_e0[frame].reshape(4, 4))[:3]*worldPixel

def convertWorldToCamera(frame,pixel):
    worldPixel = world.loc[(world['Frame']==frame) & (world['Pixel']==pixel), ['X', 'Y', 'Z', 'home']].to_numpy().reshape(4, 1)
    return (T_c_e[frame].reshape(4, 4) * T_e_e0[frame].reshape(4, 4))[:3]*worldPixel

def dehomogenize(frame,pixel):
    """convert world to camera"""
    worldPixel = world.loc[(world['Frame']==frame) & (world['Pixel']==pixel), ['X', 'Y', 'Z', 'home']].to_numpy().reshape(4, 1)
    cameraImage = (T_c_e[frame].reshape(4, 4) * T_e_e0[frame].reshape(4, 4))[:3] * worldPixel
    return K * cameraImage / cameraImage[2]

def homogenize(frame,pixel):
    """convert camera to ego"""
    cameraPixel = p.loc[(p['Frame']==frame) & (p['Pixel']==pixel),['U','V']].to_numpy()
    U = cameraPixel[0,0]
    V = cameraPixel[0,1]

    """get Y coordinate from world first frame assuming dy=0"""
    Y = convertWorldToCamera(frame,pixel)[1,0]

    Z = Y*focallength/V
    X = U*Z/focallength

    """convert to ego"""
    cameraCords =  np.hstack([[X,Y,Z],1])
    WorldCS = np.linalg.inv(T_c_e[frame].reshape(4,4))[:3]*cameraCords.reshape(4,1)

    return np.squeeze(np.array(WorldCS))



def getWorldFramePixel(frame,pixel):
    return world.loc[(world['Frame'] == frame) & (world['Pixel'] == pixel)].to_numpy()[0, 1:4]


def getCameraFramePixel(frame,pixel):
    return p.loc[(p['Frame'] == frame) & (p['Pixel'] == pixel)].to_numpy()[0, 1:4]

def getConvertedFramePixel(frame,pixel,converted):
    """get pixel by frame from converted list"""
    return np.squeeze(converted.loc[(converted['Frame'] == frame) & (converted['Pixel'] == pixel)][['X','Y','Z']].to_numpy())


def getDiffFramesWorld(frameFirst):
    if frameFirst == 0 or frameFirst>=maxFrames:
        return 0;
    data = pn.DataFrame(columns=["pixel", "dx", "dy", "dz"])
    for pixel in np.arange(59):
        first = getWorldFramePixel(frameFirst-1,pixel)
        second = getWorldFramePixel(frameFirst,pixel)
        dx, dy, dz = (second - first)
        data = data.append({"pixel": pixel, "dx": dx, "dy": dy, "dz": dz}, True)
    return data


def getDiffConvertedFrames(frameFirst,frameSecond, converted):
    data = pn.DataFrame(columns=["pixel", "dx", "dy", "dz"])
    for pixel in converted['Pixel'].unique():
        first = getConvertedFramePixel(frameFirst,pixel,converted)
        second =getConvertedFramePixel(frameSecond,pixel,converted)
        dx, dy, dz = (second - first).T
        data = data.append({"pixel": pixel, "dx": dx, "dy": dy, "dz": dz}, True)
    return data

def getConvertedFramesCamera():
    """Convert all pixels to Ego"""
    data = pn.DataFrame()
    for i,row in p.iterrows():
        pixel = row['Pixel']
        frame = row['Frame']
        x,y,z = homogenize(frame,pixel)
        data = data.append({"Pixel": pixel, "X": x, "Y": y, "Z": z,"Frame": frame}, True)
    return data


def ConvertCameraToEgo0():
    """Convert all pixels to Ego"""
    data = pn.DataFrame()
    for i,row in p.iterrows():
        pixel = row['Pixel']
        frame = row['Frame']
        CameraPixel = homogenize(frame,pixel)
        CameraPixel = np.hstack([CameraPixel, 1])
        x,y,z = np.squeeze(np.array(np.linalg.inv(T_e_e0[frame].reshape(4,4))[:3] *CameraPixel.reshape(4,1)))
        data = data.append({"Pixel": pixel, "X": x, "Y": y, "Z": z,"Frame": frame}, True)
    return data

def calculateCentroid(frame,converted):
    """calculate center of gravity"""
    return np.mean(converted.loc[converted['Frame']==frame])[['X','Y','Z']].to_numpy()

def calculateHeading(frame, converted,carLength):
    if frame == 0 or frame>=maxFrames:
        return 0;

    difference = getDiffConvertedFrames(frame-1,frame,converted);



    """get front and rear points, assuming front has biggest dx"""
    front = difference.loc[abs(difference["dx"]).idxmax()]['pixel']
    rear = difference.loc[abs(difference["dx"]).idxmin()]['pixel']

    """check if bycicle model or simple motion with fix heading"""
    if (difference[["dx","dz"]].round(3).nunique()==1).all():
        """if fix heading return 0"""
        return 0

    """get vector of plane and normalise"""
    surface = getConvertedFramePixel(frame, front,converted) - getConvertedFramePixel(frame, rear, converted)

    """get vector of steer"""
    steer = getConvertedFramePixel(frame, front,converted) - getConvertedFramePixel(frame - 1, front,converted)

    """calculate delta while looking down in y axes"""
    delta = vg.signed_angle(surface,steer,np.array([0,1,0]), units="rad")

    """if steer angle is big it can also mean fix heading"""
    if (np.degrees(delta) > 80):
        """if fix heading return 0"""
        return 0
    """calculate velocity vector at rear assuming dt =1 s"""
    velo = math.sqrt(difference.loc[difference['pixel']==rear]['dz']**2+difference.loc[difference['pixel']==rear]['dx']**2) / dt

    """vehicle length"""


    """calculate heading according to bicycle model"""
    R = carLength/math.tan(delta)
    HeadingChange = velo/R

    return math.degrees(HeadingChange)

def TransformTwoCentroids(pointsCloud):
    """"divide the clouds to two clouds according to median and calculate centroid"""
    Centroids = pn.DataFrame()
    for frame in pointsCloud['Frame'].unique():
        group = pointsCloud.loc[pointsCloud["Frame"] == frame][["X", "Z"]].__array__()
        median = np.median(group[:, 1])
        cent1 = np.mean(pointsCloud.loc[(pointsCloud['Frame'] == frame) & (pointsCloud["Z"] > median)]).to_numpy()[2:5]
        cent2 = np.mean(pointsCloud.loc[(pointsCloud['Frame'] == frame) & (pointsCloud["Z"] <= median)]).to_numpy()[2:5]

        Centroids = Centroids.append({"Pixel": 1, "X": cent1[0], "Y": cent1[1], "Z": cent1[2], "Frame": frame}, True)
        Centroids = Centroids.append({"Pixel": 2, "X": cent2[0], "Y": cent2[1], "Z": cent2[2], "Frame": frame}, True)
    return Centroids


def convertPixelCS(CS,frame,pixel):
    pixel = np.hstack([pixel, 1])
    return np.squeeze(np.array((CS[frame].reshape(4, 4))[:3] * pixel.reshape(4, 1)))

def lineraRegressionFrame(frame,converted):
    """ This function estimates the dispersion of a frame using R2 score. Trearing the pixel cloud on (X,Z) plane."""
    group = converted.loc[converted["Frame"] == frame][["X", "Z"]].__array__()
    stat = stats.linregress(group[:, 0], group[:, 1])
    return stat.rvalue
    #test.loc[test["Frame"] == frame, "Z"] = group[:, 0] * stat.slope + stat.intercept

def FurthestPointsFrame(frame,data):
    distances = squareform(pdist(data.loc[data["Frame"] == frame].to_numpy()[:, 1:4]))
    return np.max(distances)

def calculatYawFromWorld(frame):
    """calculate Yaw"""
    distances = squareform(pdist(world.loc[world["Frame"] == frame].to_numpy()[:, 1:4]))
    points = np.where(distances==np.max(distances))
    surface = getWorldFramePixel(frame, points[0][1])-getWorldFramePixel(frame, points[0][0])
    """calculate angle to Z axes"""
    return vg.signed_angle(np.array([0,0,1]),surface, np.array([0, 1, 0]), units="deg")

def mainFunction():
    """get first frame and pixel from world to compute initial yaw"""
    yaw = calculatYawFromWorld(0)

    """calculate car length from first world frame (given)"""
    carLength = FurthestPointsFrame(0,world)

    """get given camera frames and create centroids"""
    convertedEgo0 = ConvertCameraToEgo0()
    Centroids = TransformTwoCentroids(convertedEgo0)

    """get the dispersion for every frame to check relative erro by calculate r square score to etimate dispersion"""
    regressions = pn.DataFrame([round(lineraRegressionFrame(x, convertedEgo0),2) for x in np.arange(maxFrames - 1)] ,columns = ['RSqr'])

    """calculate  bars level"""
    noiseBar = np.percentile(abs(regressions['RSqr']),90);
    skipFrameBar = np.percentile(abs(regressions['RSqr']),85);

    changes=pn.DataFrame()
    for frame in np.arange(1,maxFrames):
        """calculate R square and round"""
        Rsqr = round(lineraRegressionFrame(frame,convertedEgo0),2)

        """if too noisy Rsqr<0.7 use centroids. if no noisy dont correct"""
        if ((abs(Rsqr) < skipFrameBar)):
            d_th = 0;
            isCentroid = False
            framePoints = convertedEgo0

            """get the center of gravity to calc movement"""
            x, y, z = calculateCentroid(frame, framePoints)
        elif (abs(Rsqr) < noiseBar) and (abs(Rsqr) != 1):
            framePoints = Centroids
            isCentroid = True
            """calculate heading using centroids or regular surface"""
            d_th = calculateHeading(frame, framePoints, carLength)

            """get the center of gravity to calc movement"""
            x, y, z = calculateCentroid(frame, framePoints)
        else:
            framePoints = convertedEgo0
            isCentroid = False
            """calculate heading using centroids or regular surface"""
            d_th = calculateHeading(frame,framePoints,carLength)

            """get the center of gravity to calc movement"""
            x, y, z = calculateCentroid(frame, framePoints)

        """add to yaw for heading"""
        yaw = yaw+d_th

        """calc real heading and error"""
        RealHeading = calculatYawFromWorld(frame)

        """get the center of gravity to calc movement"""
        x, y, z = calculateCentroid(frame, framePoints)  
        dx, dy, dz = [x, y, z] - calculateCentroid(frame - 1, framePoints)

        """reset every 5 frames using world"""
        if (frame%5==0):
            yaw=RealHeading
            x, y, z = calculateCentroid(frame,world)

        """calculate error"""
        if (RealHeading!=0):
            error = (yaw-RealHeading)/RealHeading;


        changes = changes.append({'Frame':frame,'dx': dx,'dy':dy,'dz': dz,'d_th': d_th,'heading': yaw, 'RealHeading': RealHeading,'error': error, 'centroid use':isCentroid ,'Rsqr': round(Rsqr,2), 'X':x,'Y':y,'Z':z},True)

changes.to_csv("onlyTargetNoise.csv")
def printDots(data,frames):
    a = convertedEgo0.loc[(convertedEgo0['Frame']) <50]
    ax = plt.axes(projection='3d')
    ax.scatter3D(a['X'], a['Y'], a['Z'], c=a['Frame'],alpha=0.1)
    ax.scatter3D(a['X'], a['Y'], a['Z'], color = 'red')
    convertedEgo0
    ax.quiver(changes['X'],changes['Y'],changes['Z'],np.sin(np.radians(changes['heading'])),changes['X']*0,np.cos(np.radians(changes['heading'])), color="red",length= 3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.invert_xaxis()
    plt.ylim(0,5)
    ax.cla()


changes['error'].mean()
