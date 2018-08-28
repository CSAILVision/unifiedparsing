
import os
import signal
from functools import partial
from multiprocessing import Pool, cpu_count

import cv2

import broden_dataset.adeseg
import broden_dataset.dtdseg
import broden_dataset.osseg
import broden_dataset.pascalseg


def generate_resized_ade20k(ade_dir):
    ds = broden_dataset.adeseg.AdeSegmentation(
        directory=ade_dir,
        version="ADE20K_2016_07_26")
    data_sets = {"ade20k": ds}
    map_in_pool(partial(resize_data_ade, verbose=True),
                all_dataset_segmentations(data_sets),
                single_process=False,
                verbose=True)


def resize_data_ade(record, verbose):
    dataset, file_index, img_path, md = record
    if verbose:
        print("{} {}".format(file_index, os.path.basename(img_path)))
    if not os.path.exists(img_path):
        return 0
    img = cv2.imread(img_path)
    h, w = img.shape[0], img.shape[1]
    max_size = 512
    if w >= h > max_size:
        h_new, w_new = max_size, round(w / float(h) * max_size)
    elif h >= w > max_size:
        h_new, w_new = round(h / float(w) * max_size), max_size
    else:
        return 0
    cv2.imwrite(img_path, cv2.resize(img, (h_new, w_new),
                                     interpolation=cv2.INTER_LINEAR))
    seg_obj_path = md["seg_filename"]
    seg = cv2.imread(seg_obj_path)
    cv2.imwrite(seg_obj_path, cv2.resize(seg, (h_new, w_new),
                                         interpolation=cv2.INTER_NEAREST))
    for i, seg_part_path in enumerate(md["part_filenames"]):
        seg = cv2.imread(seg_part_path)
        cv2.imwrite(seg_part_path, cv2.resize(seg, (h_new, w_new),
                                              interpolation=cv2.INTER_NEAREST))
    return 0


def generate_resized_os(os_dir):
    ds = broden_dataset.osseg.OpenSurfaceSegmentation(directory=os_dir)
    map_in_pool(partial(resize_data_os, verbose=True),
                all_dataset_segmentations({"opensurfaces": ds}),
                single_process=False,
                verbose=True)


def resize_data_os(record, verbose):
    dataset, file_index, filename, md = record
    img_path, seg_path = md['filename'], md['seg_filename']
    if verbose:
        print("{} {}".format(file_index, os.path.basename(img_path)))
    if not os.path.exists(img_path):
        return 0
    img = cv2.imread(img_path)
    h, w = img.shape[0], img.shape[1]
    max_size = 512
    if w >= h > max_size:
        h_new, w_new = max_size, round(w / float(h) * max_size)
    elif h >= w > max_size:
        h_new, w_new = round(h / float(w) * max_size), max_size
    else:
        return 0
    cv2.imwrite(img_path, cv2.resize(img, (w_new, h_new),
                                     interpolation=cv2.INTER_LINEAR))
    seg = cv2.imread(seg_path)
    cv2.imwrite(seg_path, cv2.resize(seg, (w_new, h_new),
                                     interpolation=cv2.INTER_NEAREST))
    return 0


def map_in_pool(fn, data, single_process=False, verbose=False):
    """
    Our multiprocessing solution; wrapped to stop on ctrl-C well.
    """
    if single_process:
        return list(map(fn, data))
    n_procs = min(cpu_count(), 32)
    original_sigint_handler = setup_sigint()
    pool = Pool(processes=n_procs, initializer=setup_sigint)
    restore_sigint(original_sigint_handler)
    try:
        if verbose:
            print('Mapping with %d processes' % n_procs)
        res = pool.map_async(fn, data)
        return res.get(31536000)
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
        raise
    finally:
        pool.close()
        pool.join()


def all_dataset_segmentations(data_sets):
    for name, ds in list(data_sets.items()):
        for i in list(range(ds.size())):
            yield (name, i, ds.filename(i), ds.metadata(i))


def setup_sigint():
    return signal.signal(signal.SIGINT, signal.SIG_IGN)


def restore_sigint(original):
    signal.signal(signal.SIGINT, original)


if __name__ == "__main__":
    root = "/afs/csail.mit.edu/u/l/liuyingcheng/code/NetDissect/dataset_toy_resized"
    generate_resized_ade20k(os.path.join(root, 'ade20k'))
    generate_resized_os(os.path.join(root, 'opensurfaces'))
