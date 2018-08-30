import numpy as np
import os
import collections
import skimage.io as skio

import random

def process_im(im):
    # processes im to [-1, 1] np.float32
    return (2. * (im / 255.) - 1.).astype(np.float32)

def deprocess_im(im):
    return (255 * (im + 1)/ 2).astype(np.uint8)

def paired_random_crop(im_a, im_b, size):
    # h x w x 3
    h, w, d = im_a.shape
    h_ = random.randint(0, h - size - 1)
    w_ = random.randint(0, w - size - 1)
    return (im_a[h_:h_ + size,
                 w_:w_ + size, :],
            im_b[h_:h_ + size,
                 w_:w_ + size, :])
    

    
    return im_as, im_bs

def unpaired_random_crop(im_a, im_b, size):
    # h x w x 3
    h, w, d = im_a.shape
    h_ = random.randint(0, h - size - 1)
    w_ = random.randint(0, w - size - 1)
    h_b = random.randint(0, h - size - 1)
    w_b = random.randint(0, w - size - 1)
    return (im_a[h_:h_ + size,
                 w_:w_ + size, :],
            im_b[h_b:h_b + size,
                 w_b:w_b + size, :])
    

    
    return im_as, im_bs


class WebCamData():
    
    def __init__(self, data_dir="./data", webcam_id="45b36e0b8480044cae88ef5b9cbff2f7"):
        self.data_dir = data_dir
        
        self.dates = os.listdir(os.path.join(data_dir, webcam_id))
        self.splits = {"train": [], # 0.7
                       "test" : [], # 0.2
                       "val"  : []} # 0.1
        
        self.dates_by_time = {}
        self.valid_examples = {}
        self.shape = None
        
        # seed: 12345
        random.seed(12345)
        for d in self.dates:
            rand = random.random()
            # hour-min
            self.dates_by_time[d] = collections.defaultdict(list)
            
            for im in os.listdir(os.path.join(self.data_dir, "%s/%s"%(webcam_id, d))):
                # if int(im.split("_")[0]) < 8 or int(im.split("_")[0]) > 15:
                #     continue
                # sorted by minutes from beginning
                self.dates_by_time[d][60. * int(im.split("_")[0]) + int(im.split("_")[1])].append(
                    os.path.join(self.data_dir, "%s/%s/%s"%(webcam_id, d, im)))
                
                if self.shape is None:
                    self.shape = skio.imread(
                        os.path.join(self.data_dir,
                                     "%s/%s/%s"%(webcam_id, d, im))).shape
            
            if rand > 0.3:
                # train
                self.splits["train"].append(d)
            elif rand > 0.1:
                # test
                self.splits["test"].append(d)
            else:
                self.splits["val"].append(d)
                
        # Unset seed
        random.seed()
        
    def get_random_pair(self, date):
        # ordered such that a occurs before b
        # returns tuple (imid_a, imid_b, ohe_time_a, ohe_time_b) by hour
        files = list(self.dates_by_time[date].keys())
        st_a = random.choice(files)
        st_b = random.choice(files)
        
        if st_a > st_b:
            # future occurs before present, swap them
            st_b, st_a = st_a, st_b
        ida = random.choice(self.dates_by_time[date][st_a])
        idb = random.choice(self.dates_by_time[date][st_b])
        
        ida_ = ida.split("/")[-1]
        idb_ = idb.split("/")[-1]
        
        minutes_a = int(ida_.split("_")[1])
        minutes_b = int(idb_.split("_")[1])
        hours_a = int(ida_.split("_")[0])
        hours_b = int(idb_.split("_")[0])
        
        time_a = np.zeros((25, )) # hours + minutes / 60
        time_b = np.zeros((25, )) # hours + minutes / 60
        
        time_a[hours_a] = 1.
        time_b[hours_b] = 1.
        
        time_a[24] = minutes_a / 60.
        time_b[24] = minutes_b / 60.
        
        return (ida, idb,
                time_a.astype(np.float32),
                time_b.astype(np.float32))
    
    def get_valid_pairs_dist_apart(self, date, dist, margin):
        # dist hours +/- margin min is valid training example
        # margin is closed
        inds = []
        snapshot_times = list(self.dates_by_time[date].keys())
        
        if len(snapshot_times) == 0:
            return inds
        
        ind_a = 0
        min_ind_b = 0
        max_ind_b = 0
        
        valid_inds = []
        
        dist = dist * 60.
        margin = margin
        
        terminate = False
        
        while not terminate:
            # mins
            if snapshot_times[ind_a] +  dist - margin > snapshot_times[min_ind_b]:
                # known min is not valid, increment min_ind_b
                # happens when new ind_a
            
                if min_ind_b + 1 >= len(snapshot_times):
                    # only empty sets now
                    break;
                    
                min_ind_b += 1
                if max_ind_b < min_ind_b:
                    max_ind_b = min_ind_b
                if len(valid_inds) >= 1:
                    valid_inds = valid_inds[1:]
                continue
                
            if min_ind_b not in valid_inds:
                valid_inds.append(min_ind_b)
            
            
            if len(snapshot_times) > max_ind_b + 1:
                # add current elements
                # don't increment
                
                # min is set, need to increment max_ind_b until the next one is out of range or outside
                if snapshot_times[ind_a] + dist + margin > snapshot_times[max_ind_b + 1]:
                    # the next maximum is in the margin
                    max_ind_b += 1
                    valid_inds.append(max_ind_b)
            
            inds.extend([(snapshot_times[ind_a], snapshot_times[vi]) for vi in valid_inds])
            
            if ind_a + 1 >= len(snapshot_times):
                break
                
            ind_a += 1
        
        # test output of algorithm
        # seen = []
        # for pair in inds:
        #     if pair in seen:
        #         1/0.
        #     seen.append(pair)
        #     diff = pair[1] - pair[0]
        #     if diff > dist + margin or diff < dist - margin:
        #         print(diff)
        #         print(pair[0], pair[1])
        #         print(dist + margin)
        #         print(dist - margin)
        #         1/0.
        return inds
    
    def batch_variable_data_with_dates(self, batch_size, paired=True,
                                       splits='train', patch_size=None):
        if patch_size is None:
            # use full-res
            crop = False
        else:
            crop = True
            
        # incldue caching
        splits_date = self.splits[splits]
        
        current, future, current_date, future_date = [], [], [], []
        for i in range(batch_size):
            d = random.choice(splits_date)
            imid_a, imid_b, time_a, time_b = self.get_random_pair(d)
            
            im_a = process_im(skio.imread(imid_a).astype(np.float32))
            im_b = process_im(skio.imread(imid_b).astype(np.float32))
        
            if crop:
                if paired:
                    im_a, im_b = paired_random_crop(im_a, im_b, patch_size)
                else:
                    im_a, im_b = unpaired_random_crop(im_a, im_b, patch_size)
            current.append(im_a)
            future.append(im_b)
            current_date.append(time_a)
            future_date.append(time_b)
            
        return (np.stack(current, axis=0),
                np.stack(future, axis=0),
                np.stack(current_date, axis=0),
                np.stack(future_date, axis=0))
            
            
    
    def batch_constant_data(self, batch_size, dist, margin=5.,
                            splits='train', patch_size=None):
        if patch_size is None:
            # use full-res
            crop = False
        else:
            crop = True

        # incldue caching
        splits_date = self.splits[splits]
        
        current, future = [], []
        
        if splits not in self.valid_examples:
            self.valid_examples[splits] = {}
        
            for d in splits_date:
                pairs = self.get_valid_pairs_dist_apart(
                            d, dist, margin)
                if len(pairs) == 0:
                    continue
                self.valid_examples[splits][d] = pairs
            
        
        for i in range(batch_size):
            d = random.choice(list(self.valid_examples[splits].keys()))
            
            im_a_id = random.choice(self.dates_by_time[d][training_pair[0]])
            im_b_id = random.choice(self.dates_by_time[d][training_pair[1]])
            im_a = process_im(skio.imread(im_a_id).astype(np.float32))
            im_b = process_im(skio.imread(im_b_id).astype(np.float32))
            
            if crop:
                im_a, im_b = paired_random_crop(im_a, im_b, patch_size)
            
            current.append(im_a)
            future.append(im_b)
            
        return np.stack(current, axis=0), np.stack(future, axis=0)
        