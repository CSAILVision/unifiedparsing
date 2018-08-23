#!/usr/bin/env mdl

""" merge multi-datasets. modified from NetDissect joinseg.py """

import csv
import json
import operator
import os
from collections import OrderedDict
from collections import namedtuple

import numpy

import broden_dataset.adeseg
import broden_dataset.dtdseg
import broden_dataset.osseg
import broden_dataset.pascalseg
from config import config


class BrodenDataset:

    def __init__(self, test_limit=None):

        """ data sets """

        # Dataset 1:    ADE20K. object, part, scene. 
        #               use resized data, use 1 level of the part seg. 
        ade = broden_dataset.adeseg.AdeSegmentation(
                directory=config.ade_dir, 
                version='ADE20K_2016_07_26')

        # Dataset 2:    Pascal context, Pascal part. object, part.
        #               collapse objectives, remove distinction between upper-arm, lower-arm, etc.
        pascal = broden_dataset.pascalseg.PascalSegmentation(
                directory=config.pascal_dir,
                collapse_adjectives=set(['left', 'right', 'front', 'back', 'upper', 'lower', 'side']))

        # Dataset 3:    dtd. texture.
        # dtd = broden_dataset.dtdseg.DtdSegmentation(directory=config.dtd_dir)

        # Dataset 4:    opensurface. material.
        #               use resized blank removed version. 
        opensurface = broden_dataset.osseg.OpenSurfaceSegmentation(directory=config.os_dir)

        self.data_sets = OrderedDict(ade20k=ade, pascal=pascal, os=opensurface)

        """ use multi source dataset """
        self.broden_dataset_info = os.path.join(config.merged_data_info_dir, 'broden_lite')
        # FIXIT(LYC):: pascal json nr part is wrong.
        # TODO(LYC):: restore complete json
        # TODO(LYC):: add validation record list
        self.record_list = {"train": [], "validation": []}
        self.record_list['train'].append(get_records(
            os.path.join(self.broden_dataset_info, "broden_ade20k_pascal_train_toy.json")))
        self.record_list['train'].append(get_records(
            os.path.join(self.broden_dataset_info, 'broden_os_train_toy.json')))

        """ recover object, part, scene, material and texture. """

        # recover names 
        self.names = {'object': [], 'part': [], 'scene': [], 'material': [], 'texture': []}
        restore_categories = ['object', 'part', 'scene', 'material', 'texture']
        for cat in restore_categories:
            self.names[cat] = [
                l.name for l in restore_csv(os.path.join(self.broden_dataset_info, '{}.csv'.format(cat)))]
        self.nr = {}
        for cat in restore_categories:
            self.nr[cat] = len(self.names[cat])

        # recover assignments 
        self.assignments = {}
        for l in restore_csv(os.path.join(self.broden_dataset_info, 'label_assignment.csv')):
            self.assignments[(l.dataset, l.category, int(l.raw_label))] = int(l.broden_label) 
        index_max = build_histogram(
                [((ds, cat), i) for ds, cat, i in list(self.assignments.keys())], max)
        self.index_mapping = dict([k, numpy.zeros(i + 1, dtype=numpy.int16)] for k, i in list(index_max.items()))
        for (ds, cat, old_index), new_index in list(self.assignments.items()):
            self.index_mapping[(ds, cat)][old_index] = new_index

        # recover object with part 
        self.object_with_part, self.object_part = [], {}
        for l in restore_csv(os.path.join(self.broden_dataset_info, 'object_part_hierarchy.csv')):
            o_l = int(l.object_label)
            self.object_with_part.append(o_l)
            self.object_part[o_l] = [int(i) for i in l.part_labels.split(';')]
        self.nr_object_with_part = len(self.object_with_part)
        
    def validation_records(self):
        raise NotImplementedError
        records = []
        assert self.names_data_source[0][0] == 'ade20k' and \
                self.names_data_source[0][1] == 'pascal'
        path = os.path.join(self.broden_dataset_info, "broden_ade20k_pascal_val.json")
        with open(path) as f:
            filelist_json = f.readlines()
        records += [json.loads(x) for x in filelist_json]
        assert self.names_data_source[1][0] == 'os'
        path = os.path.join(self.broden_dataset_info, 'broden_os_val.json')
        with open(path) as f:
            filelist_json = f.readlines()
        records += [json.loads(x) for x in filelist_json] 
        # assert self.names_data_source[2][0] == 'dtd'
        # records += self.data_sets['dtd'].validation_records() 
        return records

    def resolve_record(self, record):
        # resolve records, return: 
        #   img: 
        #   seg_obj, valid_obj: empty of valid_obj == 0
        #   seg_parts, valid_parts:  ith empty when valid_parts[i] == 0
        #   scene_label: empty when scene_label == -1
        #   texture_label: empty when texture_label == -1
        #   material, valid_mat: empty when valid_mat == 0
        # this method is specific to input pip.
       
        # decode metadata
        ds = self.data_sets[record["dataset"]]
        md = ds.metadata(record["file_index"])
        full_seg, shape = ds.resolve_segmentation(md)

        # image 
        img = ds.image_data(record["file_index"])
        seg_obj = numpy.zeros((img.shape[0], img.shape[1]), dtype=numpy.uint16)
        valid_obj = 0
        batch_seg_part = numpy.zeros((self.nr_object_with_part, img.shape[0], img.shape[1]), dtype=numpy.uint8)
        valid_part = numpy.zeros(self.nr_object_with_part, dtype=numpy.bool)
        scene_label = -1
        texture_label = -1
        seg_material = numpy.zeros((img.shape[0], img.shape[1]), dtype=numpy.uint8)
        valid_mat = 0

        # for ade20k and pascal datasets, we only decode seg_obj, seg_parts and scene. 
        if record['dataset'] == 'ade20k' or record['dataset'] == 'pascal':
            # scene 
            if record['dataset'] == 'ade20k':
                scene_label = self.index_mapping[('ade20k', 'scene')][md['scene']]

            # object
            valid_obj = 1
            seg_obj = self.index_mapping[(record["dataset"], "object")][full_seg["object"]]

            # part
            seg_part = full_seg["part"]
            if len(seg_part) == 0:
                return img, seg_obj, valid_obj, batch_seg_part, valid_part, scene_label, \
                        texture_label, seg_material, valid_mat 

            # only use first level part seg in ade20k
            if len(seg_part) != 0 and record["dataset"] == "ade20k":
                seg_part = numpy.asarray(seg_part[0], dtype=numpy.uint16)
            seg_part = self.index_mapping[(record["dataset"], "part")][seg_part]

            assert img.shape[:2] == seg_obj.shape and img.shape[:2] == seg_part.shape, \
                    "dataset: {} file_index: {}".format(record["dataset"], record["file_index"])
           
            # decode valid obj-parts. 
            for obj_part_index in range(self.nr_object_with_part):
                obj_label = self.object_with_part[obj_part_index]
                valid_part_labels = self.object_part[obj_label]
                obj_mask = (seg_obj == obj_label)
                present_part_labels = numpy.unique(seg_part * obj_mask)
                present_valid_part_labels = numpy.intersect1d(
                        valid_part_labels, present_part_labels)
                if len(present_valid_part_labels) <= 1:
                    continue 
                valid_part[obj_part_index] = True
                for v_p_label in present_valid_part_labels:
                    v_p_index = valid_part_labels.index(v_p_label)
                    batch_seg_part[obj_part_index] += numpy.asarray(
                            (seg_part == v_p_label) * obj_mask * v_p_index, dtype=numpy.uint8)

            data = {
                "img": img,
                "seg_obj": seg_obj,
                "valid_obj": valid_obj,
                "batch_seg_part": batch_seg_part,
                "valid_part": valid_part,
                "scene_label": scene_label,
                "texture_label": texture_label,
                "seg_material": seg_material,
                "valid_mat": valid_mat
            }

            return data

        # only use texture in dtd.
        elif record['dataset'] == 'dtd':
            assert len(full_seg['texture']) == 1, 'use main texture label in dtd'
            texture_label = full_seg['texture'][0]

            data = {
                "img": img,
                "seg_obj": seg_obj,
                "valid_obj": valid_obj,
                "batch_seg_part": batch_seg_part,
                "valid_part": valid_part,
                "scene_label": scene_label,
                "texture_label": texture_label,
                "seg_material": seg_material,
                "valid_mat": valid_mat
            }

            return data

        # only use material in os. 
        elif record['dataset'] == 'os':
            valid_mat = 1
            seg_material = self.index_mapping[('os', 'material')][full_seg['material']] 
            seg_material = numpy.asarray(seg_material, dtype=numpy.uint8)

            data = {
                "img": img,
                "seg_obj": seg_obj,
                "valid_obj": valid_obj,
                "batch_seg_part": batch_seg_part,
                "valid_part": valid_part,
                "scene_label": scene_label,
                "texture_label": texture_label,
                "seg_material": seg_material,
                "valid_mat": valid_mat
            }

            return data
        else:
            raise ValueError('invalid dataset name. ')


def build_histogram(pairs, reducer=operator.add):
    """Creates a histogram by combining a list of key-value pairs."""
    result = {}
    for k, v in pairs:
        if k not in result:
            result[k] = v
        else:
            result[k] = reducer(result[k], v)
    return result


def get_records(path):
    with open(path) as f:
        filelist_json = f.readlines()
    records = [json.loads(x) for x in filelist_json]
    return records


def restore_csv(csv_path):
    with open(csv_path) as f:
        f_csv = csv.reader(f)
        headings = next(f_csv)
        Row = namedtuple('Row', headings)
        lines = [Row(*r) for r in f_csv]
    return lines


broden_dataset = BrodenDataset(test_limit=None)
