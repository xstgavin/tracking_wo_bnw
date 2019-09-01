import cv2
import json
import os 
import glob
import numpy as np

from utils import get_frames

def extract_images_from_videos():
    homebrain_orgin_videos_path = '..\\..\\home_brain_pedestrain_tracking_test\\'
    homebrain_MOT_root = '..\\..\\homebrain_test\\'
    if not os.path.exists(homebrain_MOT_root):
        os.mkdir(homebrain_MOT_root)

    videos = glob.glob(homebrain_orgin_videos_path+'*.mp4')

    for video_name in videos:
        count=0
        vid_path = homebrain_MOT_root+video_name.split('\\')[-1].rstrip('.mp4')
        if not os.path.exists(vid_path):
            os.mkdir(vid_path)

        vid_path = vid_path +'\\img1\\'
        if not os.path.exists(vid_path):
            os.mkdir(vid_path)
            
        print(vid_path)

        for frame in get_frames(video_name):
            count = count +1
            img_name = '%06d.jpg'%count
            img_path = vid_path + img_name
            cv2.imwrite(img_path, frame)



def reformat_json(jdat):
    jdat_new ={}
    for elem_key in jdat.keys():
        for bb_elm in jdat[elem_key]:
            pid = bb_elm[0]
            pid_key = '%03d'%pid
            bb = [bb_elm[1],bb_elm[2],bb_elm[3],bb_elm[4]]
            if pid_key not in jdat_new.keys():
                jdat_new[pid_key]={}
                jdat_new[pid_key]['frames']=[]
                jdat_new[pid_key]['bbox']=[]
                jdat_new[pid_key]['frames'].append(int(elem_key))
                jdat_new[pid_key]['bbox'].append(bb)
            else:
                jdat_new[pid_key]['frames'].append(int(elem_key))
                jdat_new[pid_key]['bbox'].append(bb)    
    
    jdat_new_refine={}
    for key in jdat_new.keys():
        jdat_new_refine[key]={}
        jdat_new_refine[key]['frames']=[]
        jdat_new_refine[key]['bbox']=[]
        frameIdxes=jdat_new[key]['frames']
        #print(frameIdxes)
        sorted_idx=np.argsort(np.array(frameIdxes))
        #print(sorted_idx)
        frame_len = len(frameIdxes)
        #print(jdat_new[key]['frames'][sorted_idx])
        for i in range(frame_len):
            frame_idx = sorted_idx[i]
            frame = jdat_new[key]['frames'][frame_idx]
            bb = jdat_new[key]['bbox'][frame_idx]
            jdat_new_refine[key]['frames'].append(frame)
            jdat_new_refine[key]['bbox'].append(bb)
            #print(frame)
    return jdat_new_refine

def convert_jsonlabel_to_txtlabel():
    homebrain_orgin_videos_path = '..\\..\\home_brain_pedestrain_tracking_test\\'
    homebrain_MOT_root = '..\\..\\homebrain_test\\'
    if not os.path.exists(homebrain_MOT_root):
        os.mkdir(homebrain_MOT_root)

    video_labels = glob.glob(homebrain_orgin_videos_path+'*.mp4.trlabel')
    print(' convert_jsonlabel_to_txtlabel')
    for label_json_name in video_labels:
        print(label_json_name)
        label_path = homebrain_MOT_root+label_json_name.split('\\')[-1].rstrip('.mp4.trlabel')
        vid_path = homebrain_MOT_root+label_json_name.split('\\')[-1].rstrip('.mp4.trlabel')+'\\img1\\'
        frames = len(glob.glob(vid_path+'*.jpg'))

        if not os.path.exists(label_path):
            os.mkdir(label_path)

        label_path = label_path +'\\gt\\'
        if not os.path.exists(label_path):
            os.mkdir(label_path)
        wid = open(label_path+'gt.txt','w')

        fid = open(label_json_name,'r')
        jdat = json.load(fid)
        fid.close()

        jdat_new_refine=reformat_json(jdat)

        key_lens = len(jdat_new_refine.keys())
        for x in range(key_lens):
            pid_key = '%03d'%x
            frames = jdat_new_refine[pid_key]['frames']
            bboxes = jdat_new_refine[pid_key]['bbox']
            for lx  in range(len(frames)):
                frame_id = frames[lx]
                pid = x+1
                bx = bboxes[lx][0] 
                by = bboxes[lx][1] 
                bw = bboxes[lx][2] 
                bh = bboxes[lx][3] 
                line = '%d,%d,%d,%d,%d,%d,1,1,1\n'%(frame_id, pid, bx, by, bw, bh)
                wid.write(line)
        wid.close()
        # for frameIdx in range(frames):
        #     frame_str = '%06d'%(frameIdx+1)
        #     if key not 

    print("NOT implemented")


if __name__=='__main__':
    #extract_images_from_videos()
    convert_jsonlabel_to_txtlabel()