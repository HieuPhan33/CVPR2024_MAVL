import numpy as np
import json
# List of most common normal observations
NORM_OBS = ['normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free']

# exclude
EXCLUDED_OBS = ['none', 'unchanged', 'change', 'great', 'similar', 'large', 'small', 'moderate', 'mild',
                'median', 'decrease', 'bad', 'more', 'constant', 'worsen', 'new', 'improve',
                'status', 'position', 'sternotomy', 'cabg', 'replacement', 'postoperative', 'assessment',
                'patient']

# top 90% abnormal observations
ABNORM_OBS = ['effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation', 'process',
              'abnormality', 'enlarge', 'tip', 'low', 'pneumonia', 'line', 'congestion', 'catheter', 'cardiomegaly',
              'fracture', 'air', 'tortuous', 'lead', 'disease', 'calcification', 'prominence', 'device',
              'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire', 'fluid', 'degenerative',
              'pacemaker', 'thicken', 'marking', 'scar', 'hyperinflate', 'blunt', 'loss', 'widen', 'collapse',
              'density', 'emphysema', 'aerate', 'mass', 'crowd', 'infiltrate', 'obscure', 'deformity', 'hernia',
              'drainage', 'distention', 'shift', 'stent', 'pressure', 'lesion', 'finding', 'borderline', 'hardware',
              'dilation', 'chf', 'redistribution', 'aspiration']

# final row and column names in adjacent matrix
LANDMARK_NAME = ['trachea', 'left_hilar', 'right_hilar', 'hilar_unspec', 'left_pleural', 'right_pleural',
                 'pleural_unspec', 'heart_size', 'heart_border', 'left_diaphragm', 'right_diaphragm',
                 'diaphragm_unspec', 'retrocardiac', 'lower_left_lobe', 'upper_left_lobe', 'lower_right_lobe',
                 'middle_right_lobe', 'upper_right_lobe', 'left_lower_lung', 'left_mid_lung', 'left_upper_lung',
                 'left_apical_lung', 'left_lung_unspec', 'right_lower_lung', 'right_mid_lung', 'right_upper_lung',
                 'right_apical_lung', 'right_lung_unspec', 'lung_apices', 'lung_bases', 'left_costophrenic',
                 'right_costophrenic', 'costophrenic_unspec', 'cardiophrenic_sulcus', 'mediastinal', 'spine',
                 'clavicle', 'rib', 'stomach', 'right_atrium', 'right_ventricle', 'aorta', 'svc', 'interstitium',
                 'parenchymal', 'cavoatrial_junction', 'cardiopulmonary', 'pulmonary', 'lung_volumes',
                 'unspecified', 'other']

'''
free: free air, no congestion
tip: catheter's tip
'''
OBSERVATION_CLASS = ['normal', 'clear', 'sharp', 'sharply', 'unremarkable', 'intact', 'stable', 'free', 'effusion',
                     'opacity', 'pneumothorax', 'edema', 'atelectasis', 'tube', 'consolidation', 'process',
                     'abnormality', 'enlarge', 'tip', 'low', 'pneumonia', 'line', 'congestion', 'catheter',
                     'cardiomegaly', 'fracture', 'air', 'tortuous', 'lead', 'disease', 'calcification',
                     'prominence', 'device', 'engorgement', 'picc', 'clip', 'elevation', 'expand', 'nodule', 'wire',
                     'fluid', 'degenerative', 'pacemaker', 'thicken', 'marking', 'scar', 'hyperinflate', 'blunt',
                     'loss', 'widen', 'collapse', 'density', 'emphysema', 'aerate', 'mass', 'crowd', 'infiltrate',
                     'obscure', 'deformity', 'hernia', 'drainage', 'distention', 'shift', 'stent', 'pressure',
                     'lesion', 'finding', 'borderline', 'hardware', 'dilation', 'chf', 'redistribution',
                     'aspiration', 'tail_abnorm_obs', 'excluded_obs']

target = 'redistribution'
location = 'stomach'
np_path = '/data/VLM/MedKLIP/PreTrain_MedKLIP/data_file/landmark_observation_adj_mtx.npy'
rad_graph_results = np.load(np_path) # N_report, 51, 75
data = json.load(open('/data/VLM/MedKLIP/PreTrain_MedKLIP/data_file/rad_graph_metric_train.json', 'r'))
report_idx, anatomy_idx = np.where(rad_graph_results[:, :, OBSERVATION_CLASS.index(target)] == 1)
# report_idx, obs_idx = np.where(rad_graph_results[:, LANDMARK_NAME.index(location), :] == 0)


idxes = [0,1,2,3,4,5]
for idx in idxes:
    report_id = report_idx[idx]
    anatomy_id = anatomy_idx[idx]
    # obs_id = obs_idx[idx]
    files = [k for k,v in data.items() if v['labels_id'] == report_id]
    for f in files:
        f = f.replace('/remote-home/share/medical/public/MIMIC-CXR-JPG/MIMIC-CXR/small', '/data/VLM/data/2019.MIMIC-CXR-JPG/2.0.0')
        f = '/'.join(f.split('/')[:-1])
        content = open(f'{f}.txt', 'r').read()
        print(f"Finding {target} --- Anatomy {LANDMARK_NAME[anatomy_id]}")
        #print(f"Anatomy {location} --- Observation {OBSERVATION_CLASS[obs_id]}")
        print(content)

# report_idxes = [report_idx[i] for i in idxes]
# files  = {report_idx.get() for k,v in data.items() if v['labels_id'] in report_idxes}
# for f in files:
#     f = f.replace('/remote-home/share/medical/public/MIMIC-CXR-JPG/MIMIC-CXR/small', '/data/VLM/data/2019.MIMIC-CXR-JPG/2.0.0')
#     f = '/'.join(f.split('/')[:-1])
#     content = open(f'{f}.txt', 'r').read()
#     print(content)
