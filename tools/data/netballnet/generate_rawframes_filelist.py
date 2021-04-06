import json
import os
import csv
import os.path as osp

data_file = '../../../data/NetballNet'
video_list = f'{data_file}/video_info.csv'
anno_file = f'{data_file}/nnet_anno_action.json'
rawframe_dir = f'{data_file}/rawframes'
video_info_file = f'{data_file}/video_info.csv'
action_name_list = 'action_name.csv'

train_rawframe_dir = rawframe_dir
val_rawframe_dir = rawframe_dir

json_file = f'{data_file}/netball_net_v1.json'

#read video info csv
with open(video_info_file) as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    video_info_data = list(csv_reader)
csv_file.close()

frame_count_dict = []
for entry in video_info_data:
    single_entry = {
        'video_name' : entry[0] ,
        'num_frames' : int(entry[1])
    }
    frame_count_dict.append(single_entry)

def generate_rawframes_filelist():

    def get_frame_count(video_name):
        for item in frame_count_dict:
            if item['video_name'] == video_name:
                return item['num_frames']

    load_dict = json.load(open(json_file))

    anet_labels = open(action_name_list).readlines()
    anet_labels = [x.strip() for x in anet_labels[1:]]

    train_dir_list = [
        osp.join(train_rawframe_dir, x) for x in os.listdir(train_rawframe_dir)
    ]
    val_dir_list = [
        osp.join(val_rawframe_dir, x) for x in os.listdir(val_rawframe_dir)
    ]

    def simple_label(anno):
        label = anno[0]['label']
        return anet_labels.index(label)

    def count_frames(dir_list, video):
        for dir_name in dir_list:
            if video in dir_name:
                video_name = video.split('/')[-1]
                return osp.basename(dir_name), get_frame_count(video_name)
                # return osp.basename(dir_name), len(os.listdir(dir_name))
        return None, None

    database = load_dict['database']
    training = {}
    validation = {}
    key_dict = {}

    for k in database:
        data = database[k]
        subset = data['subset']

        if subset in ['training', 'validation']:
            annotations = data['annotations']
            label = simple_label(annotations)
            if subset == 'training':
                dir_list = train_dir_list
                data_dict = training
            else:
                dir_list = val_dir_list
                data_dict = validation

        else:
            continue

        gt_dir_name, num_frames = count_frames(dir_list, k)
        if gt_dir_name is None:
            continue
        data_dict[gt_dir_name] = [num_frames, label]
        key_dict[gt_dir_name] = k

    train_lines = [
        k + ' ' + str(training[k][0]) + ' ' + str(training[k][1])
        for k in training
    ]
    val_lines = [
        k + ' ' + str(validation[k][0]) + ' ' + str(validation[k][1])
        for k in validation
    ]

    with open(osp.join(data_file, 'nnet_train_video.txt'), 'w') as fout:
        fout.write('\n'.join(train_lines))
    with open(osp.join(data_file, 'nnet_val_video.txt'), 'w') as fout:
        fout.write('\n'.join(val_lines))

    def clip_list(k, anno, video_anno):
        duration = anno['duration']
        num_frames = video_anno[0]
        fps = num_frames / duration
        segs = anno['annotations']
        lines = []
        for seg in segs:
            segment = seg['segment']
            label = seg['label']
            label = anet_labels.index(label)
            start, end = int(segment[0] * fps), int(segment[1] * fps)
            if end > num_frames - 1:
                end = num_frames - 1
            newline = f'{k} {start} {end - start + 1} {label}'
            lines.append(newline)
        return lines

    train_clips, val_clips = [], []
    for k in training:
        train_clips.extend(clip_list(k, database[key_dict[k]], training[k]))
    for k in validation:
        val_clips.extend(clip_list(k, database[key_dict[k]], validation[k]))

    with open(osp.join(data_file, 'nnet_train_clip.txt'), 'w') as fout:
        fout.write('\n'.join(train_clips))
    with open(osp.join(data_file, 'nnet_val_clip.txt'), 'w') as fout:
        fout.write('\n'.join(val_clips))


if __name__ == '__main__':
    generate_rawframes_filelist()
