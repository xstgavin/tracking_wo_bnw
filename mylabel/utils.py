import glob
import cv2
import numpy as np
import json

color = {}
color['0']=(255,0,0)
color['1']=(255,255,0)
color['2']=(255,0,255)
color['3']=(0,255,0)
color['4']=(0,255,255)
color['5']=(0,0,255)

def get_frames(video_name):
    cap = cv2.VideoCapture(video_name)
    while True:
        ret, frame = cap.read()
        if ret:
            yield frame
        else:
            break

def reformat_json(jdat):
    # reformat json data from by tracking id to by frame
    # 
    frameInfo={}
    for elm in jdat['ids']:
        tid = elm['id']
        for bbox in elm['trackInfo']:
            frameIdx = '%06d'%bbox[0]
            if frameIdx not in frameInfo.keys():
                frameInfo[frameIdx]=[]
            frameInfo[frameIdx].append([tid,bbox[1],bbox[2],bbox[3],bbox[4]])
    return frameInfo

def draw_track_path(frame,bbox,tid):
    pt0 = (bbox[0],bbox[1])
    pt1 = (bbox[0]+bbox[2],bbox[1]+bbox[3])
    print(color[str(tid)])
    cv2.rectangle(frame,pt0,pt1,color[str(tid)],2)
    cv2.putText(frame,'%03d'%tid,pt0,cv2.FONT_HERSHEY_SIMPLEX,0.2,color[str(tid+1)],1)
    return frame

def set_bbox(window_name,frame,idx_str, tid,jdat):
    
    nw_box = cv2.selectROI(window_name, frame, False, False)
    is_set = False
    if idx_str not in jdat.keys():
        jdat[idx_str] = []
        b_data = [tid,nw_box[0],nw_box[1],nw_box[2],nw_box[3]]
        jdat[idx_str].append(b_data)
    for bbox in jdat[idx_str]:
        if  bbox[0] != tid:
            continue
        else:
            bbox[1] = nw_box[0]
            bbox[2] = nw_box[1]
            bbox[3] = nw_box[2]
            bbox[4] = nw_box[3]
            is_set = True
    if not is_set:
        jdat[idx_str].append([tid,nw_box[0],nw_box[1],nw_box[2],nw_box[3]])
    

def get_labels(path, extension='pysot',fmt='old'):
    label_list = glob.glob(path+'*.'+extension)
    for label_name in label_list:
        print(label_name)
        fid = open(label_name,'r')
        jdat = json.load(fid)
        fid.close()
        if fmt == 'old':
            jdat=reformat_json(jdat)
        
        video_name = label_name.rstrip('.'+extension)
        print(video_name)
        frameIdx=0
        #idInfo = jdat['ids'][0]
        window_name='xx'
        for frame in get_frames(video_name):
            framebk = frame.copy()
            frameIdx = frameIdx +1
            idx_str = '%06d'%frameIdx
            if idx_str not in jdat.keys():
                cv2.imshow(window_name,frame)
                cv2.waitKey(30)
                x=input('change?')
                if x=='':
                    continue
                else:
                    tid = int(x)
                    set_bbox(window_name,frame,idx_str,tid,jdat)
                    for bbinfo in jdat[idx_str]:
                        tid = bbinfo[0]
                        bbox=np.array([bbinfo[1],bbinfo[2],bbinfo[3],bbinfo[4]])
                        frame=draw_track_path(frame,bbox,tid)
                    cv2.imshow(window_name,frame)
                    cv2.waitKey(30)
                    continue
            for bbinfo in jdat[idx_str]:
                tid = bbinfo[0]
                bbox=np.array([bbinfo[1],bbinfo[2],bbinfo[3],bbinfo[4]])
                frame=draw_track_path(frame,bbox,tid)
            cv2.imshow(window_name,frame)
            cv2.waitKey(30)
            x=input('change?')
            if x=='':
                continue
            else:
                tid = int(x)
                set_bbox(window_name,frame,idx_str,tid,jdat)
                for bbinfo in jdat[idx_str]:
                    tid = bbinfo[0]
                    bbox=np.array([bbinfo[1],bbinfo[2],bbinfo[3],bbinfo[4]])
                    frame=draw_track_path(frame,bbox,tid)
                cv2.imshow(window_name,frame)
                cv2.waitKey(30)
        #print(jdat)
        fid = open(label_name,'w')
        json.dump(jdat,fid,indent=4)
        fid.close()
    
def test():
    #get_labels('../../data_real/videos/',extension='pysot_crt',fmt='new')
    get_labels('../../data_real/videos/',extension='pysot',fmt='new')

if __name__ == "__main__":
    test()
