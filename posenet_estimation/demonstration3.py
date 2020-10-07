import argparse
from functools import partial
import re
import time
import datetime
import os

import numpy as np
from PIL import Image
import cv2
# import svgwrite
# import gstreamer

from myline import LINENotifyBot
import threading
import subprocess
import json
import math

from pose_engine import PoseEngine

#choice line
EDGES = (
    ('nose', 'left eye'),
    ('nose', 'right eye'),
    ('nose', 'left ear'),
    ('nose', 'right ear'),
    ('left ear', 'left eye'),
    ('right ear', 'right eye'),
    ('left eye', 'right eye'),
    ('left shoulder', 'right shoulder'),
    ('left shoulder', 'left elbow'),
    ('left shoulder', 'left hip'),
    ('right shoulder', 'right elbow'),
    ('right shoulder', 'right hip'),
    ('left elbow', 'left wrist'),
    ('right elbow', 'right wrist'),
    ('left hip', 'right hip'),
    ('left hip', 'left knee'),
    ('right hip', 'right knee'),
    ('left knee', 'left ankle'),
    ('right knee', 'right ankle'),
)


path_record = 'record/test2.txt'
# time get
ut = time.time()
dt = datetime.datetime.now()
ut = math.floor(ut)
dt_str = str(dt) + '\n'

with open(path_record,mode='w') as f:
    f.write(dt_str)




def shadow_text(img, x, y, text, font_size=0.5):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0,0,255), 1, cv2.LINE_AA)
    
    
def face_direct(xys):
    color = (0,255,0)
    if   'nose'in xys and 'left eye' in xys and 'right eye' in xys:
        color = (255,0,255)
    return color

def rightarm_detect(pose):
    rw =pose.keypoints['right wrist']
    rs = pose.keypoints['right shoulder']
  #  ns = pose.keypoints['nose']

    THR = 0.5
    if( rw.score > THR and rs.score > THR):
        return(rw.yx[0] < rs.yx[0])
       
def draw_pose(img, pose, threshold=0.2):
    xys = {}
    for label, keypoint in pose.keypoints.items():
        if keypoint.score < threshold: continue
        xys[label] = (int(keypoint.yx[1]), int(keypoint.yx[0]))
        img = cv2.circle(img, (int(keypoint.yx[1]), int(keypoint.yx[0])), 5, (0, 255, 0), -1)

    #print("_--------",xys.get('left hip'))
        """
    for a, b in EDGES:
        if a not in xys or b not in xys: continue
        ax, ay = xys[a]
        bx, by = xys[b]
        color = (0,255,0)

        color = face_direct(xys)

        img = cv2.line(img, (ax, ay), (bx, by), color, 2)"""
        
            
def squad(pose):
    rh = pose.keypoints['right hip']
    lh = pose.keypoints['left hip']
    rk = pose.keypoints['right knee']
    lk = pose.keypoints['left knee']
    ra = pose.keypoints['right ankle']
    la = pose.keypoints['left ankle']
    
    THR = 0.5
    #if(rw.score > THR and lw.score > THR and rs.score > THR and ls.score > THR and re.score > THR \
       #and le.score > THR and rh.score > THR and lh.score > THR and rk.score > THR and lk.score > THR and ra.score > THR and la.score > THR):
    if(rk.score > THR and lk.score > THR and rh.score > THR and lh.score > THR):
        
        a = math.floor(((ra.yx[0] - rk.yx[0])**2) + ((ra.yx[1] - rk.yx[1])**2))
        criteria = math.sqrt(a)
        ratio = criteria * 3 / 10
        
        if(rh.yx[0] > (rk.yx[0] - ratio) and lh.yx[0] > (lk.yx[0] - ratio)):
            print('OK')    
                
def neck(pose):
    reye = pose.keypoints['right eye']
    leye = pose.keypoints['left eye']
    rear = pose.keypoints['right ear']
    lear = pose.keypoints['left ear']
    nose = pose.keypoints['nose']
    
    THR = 0.5
    if(reye.score > THR and leye.score > THR and rear.score > THR and lear.score > THR and nose.score > THR):
        """
        print('--reye--')
        print(reye.yx[0], reye.yx[1])
        print('--rear--')
        print(rear.yx[0], rear.yx[1])
        print('--nose--')
        print(nose.yx[0], nose.yx[1])"""
        
        criteria = reye.yx[1] - rear.yx[1]
        ratio_up = criteria * 3 / 2
        ratio_down = criteria * 1.2
        a = math.floor(((nose.yx[0] - rear.yx[0])**2) + ((nose.yx[1] - rear.yx[1])**2))
        criteria_2 = math.sqrt(a)
        ratio_yoko1 = criteria_2 / 10 * -1
        ratio_yoko2 = criteria_2 * 3 * 0.02
        ratio_yoko3 = criteria_2 * 3 / 10
        
        if((rear.yx[0] - nose.yx[0]) > ratio_up and (lear.yx[0] - nose.yx[0]) > ratio_up):
            print('up')
        elif(nose.yx[0] - rear.yx[0] > ratio_down and nose.yx[0] - lear.yx[0] > ratio_down):
            print('down')
        elif(rear.yx[0] - reye.yx[0] > ratio_yoko1 and rear.yx[0] - reye.yx[0] < ratio_yoko2):
            if(lear.yx[0] - nose.yx[0] > ratio_yoko3):
                print('left')
        elif(lear.yx[0] - leye.yx[0] > ratio_yoko1 and lear.yx[0] - leye.yx[0] < ratio_yoko2):
            if(rear.yx[0] - nose.yx[0] > ratio_yoko3):
                print('right')

def pose_record(pose):
    # convert into string
    
    nose = pose.keypoints['nose']
    left_eye = pose.keypoints['left eye']
    right_eye = pose.keypoints['right eye']
    left_ear = pose.keypoints['left ear']
    right_ear = pose.keypoints['right ear']
    left_shoulder = pose.keypoints['left shoulder']
    right_shoulder = pose.keypoints['right shoulder']
    left_elbow = pose.keypoints['left elbow']
    right_elbow = pose.keypoints['right elbow']
    left_wrist = pose.keypoints['left wrist']
    right_wrist = pose.keypoints['right wrist']
    left_hip = pose.keypoints['left hip']
    right_hip = pose.keypoints['right hip']
    left_knee = pose.keypoints['left knee']
    right_knee = pose.keypoints['right knee']
    left_ankle = pose.keypoints['left ankle']
    right_ankle = pose.keypoints['right ankle']
    
    new_ut = time.time()
    new_dt = datetime.datetime.now()
    new_ut = math.floor(new_ut)
    new_dt_str = str(new_dt)
    
    with open(path_record,mode='r') as f:
            lines = f.read().splitlines()
            last_line = lines[-1]
            last_line = last_line[17:19]
    
    if(last_line != new_dt_str[17:19]):
        
        if ((int(new_ut) - int(ut)) % 5 == 0):
            s = str(new_dt) + ':' + ' (' + str(math.floor(nose.yx[0])) + ',' + str(math.floor(nose.yx[1])) + ')  ' \
                + '(' + str(math.floor(left_eye.yx[0])) + ',' + str(math.floor(left_eye.yx[1])) + ')  ' + '(' + str(math.floor(right_eye.yx[0])) + ',' + str(math.floor(right_eye.yx[1])) + ')  ' \
                + '(' + str(math.floor(left_ear.yx[0])) + ',' + str(math.floor(left_ear.yx[1])) + ')  ' + '(' + str(math.floor(right_ear.yx[0])) + ',' + str(math.floor(right_ear.yx[1])) + ')  ' \
                + '(' + str(math.floor(left_shoulder.yx[0])) + ',' + str(math.floor(left_shoulder.yx[1])) + ')  ' + '(' + str(math.floor(right_shoulder.yx[0])) + ',' + str(math.floor(right_shoulder.yx[1])) + ')  ' \
                + '(' + str(math.floor(left_elbow.yx[0])) + ',' + str(math.floor(left_elbow.yx[1])) + ')  ' + '(' + str(math.floor(right_elbow.yx[0])) + ',' + str(math.floor(right_elbow.yx[1])) + ')  ' \
                + '(' + str(math.floor(left_wrist.yx[0])) + ',' + str(math.floor(left_wrist.yx[1])) + ')  ' + '(' + str(math.floor(right_wrist.yx[0])) + ',' + str(math.floor(right_wrist.yx[1])) + ')  ' \
                + '(' + str(math.floor(left_hip.yx[0])) + ',' + str(math.floor(left_hip.yx[1])) + ')  ' + '(' + str(math.floor(right_hip.yx[0])) + ',' + str(math.floor(right_hip.yx[1])) + ')  ' \
                + '(' + str(math.floor(left_knee.yx[0])) + ',' + str(math.floor(left_knee.yx[1])) + ')  ' + '(' + str(math.floor(right_knee.yx[0])) + ',' + str(math.floor(right_knee.yx[1])) + ')  ' \
                + '(' + str(math.floor(left_ankle.yx[0])) + ',' + str(math.floor(left_ankle.yx[1])) + ')  ' + '(' + str(math.floor(right_ankle.yx[0])) + ',' + str(math.floor(right_ankle.yx[1])) + ')\n'
            
            with open(path_record,mode='a') as f:
                f.write(s)
            



def posecamera(e):

    e.clear()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mirror', help='flip video horizontally', action='store_true')
    parser.add_argument('--model', help='.tflite model path.', required=False)
    parser.add_argument('--res', help='Resolution', default='640x480',
                        choices=['480x360', '640x480', '1280x720'])
    parser.add_argument('--videosrc', help='Which video source to use (WebCam number or video file path)', default='0')
    # parser.add_argument('--h264', help='Use video/x-h264 input', action='store_true')
    args = parser.parse_args()

    default_model = 'models/posenet_mobilenet_v1_075_%d_%d_quant_decoder_edgetpu.tflite'
    if args.res == '480x360':
        src_size = (640, 480)
        appsink_size = (480, 360)
        model = args.model or default_model % (353, 481)
    elif args.res == '640x480':
        src_size = (640, 480)
        appsink_size = (640, 480)
        model = args.model or default_model % (481, 641)
    elif args.res == '1280x720':
        src_size = (1280, 720)
        appsink_size = (1280, 720)
        model = args.model or default_model % (721, 1281)

    print('Loading model: ', model)
    engine = PoseEngine(model, mirror=args.mirror)
    # engine = PoseEngine(model)


    last_time = time.monotonic()
    n = 0
    sum_fps = 0
    sum_process_time = 0
    sum_inference_time = 0

    width, height = src_size

    isVideoFile = False
    frameCount = 0
    maxFrames = 0
    img1 = cv2.imread('gu.jpg')
    img2 = cv2.imread('tyoki.jpg')
    img3 = cv2.imread('pa.jpg')
    img11 = cv2.resize(img1, (100, 100))
    img21 = cv2.resize(img2, (100, 100))
    img31 = cv2.resize(img3, (100, 100))
    janken = 0

    # VideoCapture init
    videosrc = args.videosrc
    if videosrc.isdigit():
        videosrc = int(videosrc)
    else:
        isVideoFile = os.path.exists(videosrc)

    print("Start VideoCapture")
    cap = cv2.VideoCapture(videosrc)
        
    
    
    if cap.isOpened() == False:
        print('can\'t open video source \"%s\"' % str(videosrc))
        return;

    print("Open Video Source")
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if isVideoFile:
        maxFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
    guuu = 0
    tyokii = 0
    paaa = 0

    try:
        i=0
        pose_ary = {}
        flag = False
        while (True
            ):
            ret, frame = cap.read()
            if ret == False:
                print('can\'t read video source')
                break;

           # frame = cv2.imread('./images/output'+str(i)+'.jpg')
            rgb = frame[:,:,::-1]

#             nonlocal n, sum_fps, sum_process_time, sum_inference_time, last_time
            start_time = time.monotonic()
            # image = Image.fromarray(rgb)
            outputs, inference_time = engine.DetectPosesInImage(rgb)
            
            end_time = time.monotonic()
     #      cv2.imshow('TEST PoseNet - OpenCV', frame)

            n += 1
            sum_fps += 1.0 / (end_time - last_time)
            sum_process_time += 1000 * (end_time - start_time) - inference_time
            sum_inference_time += inference_time
            last_time = end_time
            text_line = 'PoseNet: %.1fms Frame IO: %.2fms TrueFPS: %.2f Nposes %d' % (
                sum_inference_time / n, sum_process_time / n, sum_fps / n, len(outputs)
            )
#            print(text_line)

            # crop image
            imgDisp = frame[0:appsink_size[1], 0:appsink_size[0]].copy()
##            cv2.imshow('TEST2 PoseNet - OpenCV', frame)

            if args.mirror == True:
                imgDisp = cv2.flip(imgDisp, 1)

            shadow_text(imgDisp, 10, 20, text_line)#text_line in imgDisp  here 
            
            for pose in outputs:#-----------------------------------------------------------------------------------------------------------------------------------------
                draw_pose(imgDisp, pose)
                #squad(pose)
                #neck(pose)
                pose_record(pose)

            if i >= 40:
                i = 0
                flag = False
            if flag: i+=1
#            print(i)
            e.clear()
    
            #cv2.imshow('PoseNet - OpenCV', imgDisp)
            
            
            imgDisp_resize = imgDisp.copy()
            if(janken == 1):
                imgDisp_resize[100:200, 100:200] = img11
            elif(janken == 2):
                imgDisp_resize[100:200, 100:200] = img21
            elif(janken == 3):
                imgDisp_resize[100:200, 100:200] = img31
            
            cv2.imshow('TEST', imgDisp_resize)
            
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if isVideoFile:
                frameCount += 1
                # check frame count
                if frameCount >= maxFrames:
                    # rewind video file
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frameCount = 0


    except Exception as ex:
        raise ex
    finally:
        cv2.destroyAllWindows()
        cap.release()


def lineNtf(e):

    """
    myline = LINENotifyBot(access_token = token)
    azure_url = "https://hst-face.cognitiveservices.azure.com"
    azure_pg = "hst-test"
    """

    while(True):
        e.wait()
        """
        try:
            output_js = subprocess.check_output(["/usr/local/bin/dotnet" ,"/home/pi/netcoreapp3.1/AzureFaceAuthorizer.dll" ,azure_key, azure_url ,"1", azure_pg,"test.jpg"]).decode()
            output_dic = json.loads(output_js)
            message = output_dic['error']
            if message == None:
                message = output_dic['name']
        except subprocess.CalledProcessError as e:
            message = 'Azure Failed'
        message = 'detected'
        myline.send(message=message,image='test.jpg')
        """
        print('message sent')

def sound(e):
    while(True):
        e.wait()
        subprocess.run(['aplay','file_20131208_Cursor3.wav','-q'])
        print('sound played')

if __name__ == '__main__':
    e = threading.Event()
    e.clear()
    thread_posecamera = threading.Thread(target=posecamera,args=(e,))
#    thread_lineNtf = threading.Thread(target=lineNtf, args=(e,))
    thread_sound = threading.Thread(target=sound, args=(e,))


    thread_posecamera.start()
#    thread_lineNtf.start()
    thread_sound.start()

    print("ALL THREAD STARTED")




