#!/usr/bin/env python2.7
#
# OpenSurfaces download helper script (Version 1.0).
# ==================================================
#
# This script does the following:
#
#   1. Converts data from JSON to a more convenient CSV format.  Note that
#      some data is not converted to keep things simple (e.g. response times,
#      mturk usernames, multiple submissions, incorrect submissions, etc.).
#
#   2. Download all high-resolution photos from our server.
#
#   3. Render "scene parsing" style images for the photos where each pixel
#      encodes the material/object/scene categories.
#
#   4. Crop out each shape from the photo using its polygon and then rectify
#      it using its normal.
#
# The script expects that all JSON files are contained a local directory named
# "opensurfaces/".  To modify which steps are performed, see the script options
# below.
#
# Package dependencies:
#
#    pillow>=2.5.1 or PIL>=1.1.7
#    numpy>=1.8.0
#    colormath>=1.0.8
#
# Run the script with the command:
#
#    python process_opensurfaces_release_0.py
#
# The script generates the following directories and files:
#
#    photos/            original high-resolution photos, with the OpenSurfaces
#                       ID as the filename
#
#    photos-labels/     substance (aka material) and name (aka object) labels
#                       for that photo, with the substance encoded in the red
#                       channel, name encoded in the green channel, and scene
#                       encoded in the blue channel
#
#    shapes-cropped/    cropped polygon for each shape (see CROP_SHAPES flag)
#
#    shapes-rectified/  rectified texture for each shape using the normal (see
#                       RECTIFY_SHAPES flag)
#
#    shapes-bsdf-blobs/ blob matched to the shape (Ward BRDF)
#
#    photos.csv    photos table: scene category, whitebalance,
#                  field of view, and license info, for each photo
#
#    shapes.csv    shapes table: substance (aka material), name (aka object),
#                  planarity, surface normal, diffuse albedo and gloss cd
#                  (contrast and distinctness of image), and the photo id, for
#                  each shape.
#
#                  Note regarding diffuse albedo: Since the diffuse albedo
#                  assumes a single lighting environment, the scale of the
#                  diffuse albedo is incorrect and only the chromaticity of two
#                  albedos can be reliably compared.
#
#    label-substance-colors.csv   the red color encoding used for photos-labels/
#    label-name-colors.csv        the green color encoding used for photos-labels/
#    label-scene-colors.csv       the blue color encoding used for photos-labels/
#

#### SCRIPT OPTIONS ####

# if True, download the original high-resolution photo for all images in the
# database
DOWNLOAD_PHOTO_IMAGES = True

# if 'True', render the labels to an image, encoding the substances (materials)
# in the red channel and the names (objects) in the green channel
RENDER_PHOTO_LABELS = True

# resolution used to render the label images
PHOTO_LABELS_HEIGHT = 1024

# if True, crop out each shape's pixels using its polygon
CROP_SHAPES = True

# if True, rectify textures for each shape
RECTIFY_SHAPES = True

# if True, download BRDF blob images for each shape
DOWNLOAD_BSDF_BLOBS = True


###############################################################################

OPENSURFACES_STATIC_URL_ROOT = 'http://labelmaterial.s3.amazonaws.com'


import csv
import json
import math
import multiprocessing
import os
import sys
import urllib.request, urllib.error, urllib.parse

from PIL import Image, ImageDraw
import numpy as np


def process_photos(pool):
    """ Download all photos """

    print('load scene names...')
    scenes = {p['pk']: p['name'] for p in load_json('photos.photoscenecategory.json')}
    flickr_users = {p['pk']: p['username'] for p in load_json('photos.flickruser.json')}
    licenses = {p['pk']: p for p in load_json('licenses.license.json')}

    print('create directory')
    if DOWNLOAD_PHOTO_IMAGES:
        if not os.path.exists('photos'):
            os.makedirs('photos')

    print('processing photos')
    with open('photos.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow((
            'photo_id',
            'scene_category_name',
            'scene_category_score',
            'whitebalanced',
            'whitebalanced_score',
            'fov',
            'flickr_url',
            'license_name',
            'license_is_creative_commons',
        ))

        photos = load_photos()
        for i, p in enumerate(photos):
            flickr_user = flickr_users.get(p['flickr_user'], None)
            if flickr_user:
                flickr_url = "http://www.flickr.com/photos/%s/%s/" % (
                    flickr_user, p['flickr_id'])
            else:
                flickr_url = None

            license = licenses.get(p['license'], None)
            if license:
                license_name = license['name']
                license_cc = license['creative_commons']
            else:
                license_name = "All Rights Reserved"
                license_cc = False

            writer.writerow((
                p['pk'],
                scenes.get(p['scene_category'], None),
                p['scene_category_correct_score'],
                p['whitebalanced'],
                p['whitebalanced_score'],
                p['fov'],
                flickr_url,
                license_name,
                license_cc,
            ))

            if DOWNLOAD_PHOTO_IMAGES:
                filename = os.path.join('photos', '%s.jpg' % p['pk'])
                url = '%s/%s' % (OPENSURFACES_STATIC_URL_ROOT, p['image_orig'])
                progress_str = '%s/%s' % (i, len(photos))
                pool.apply_async(download_image, (progress_str, filename, url))


def process_shapes(pool):
    """ Process all material shapes """

    print('load substance names...')
    substances = {p['pk']: p['name'] for p in load_json('shapes.shapesubstance.json') if not p['fail']}
    names = {p['pk']: p['name'] for p in load_json('shapes.shapename.json') if not p['fail']}
    bsdfs = {p['pk']: p for p in load_json('shapes.shapebsdflabel_wd.json')}
    normals = {p['pk']: p for p in load_json('shapes.shaperectifiednormallabel.json')}

    if CROP_SHAPES:
        if not os.path.exists('shapes-cropped'):
            os.makedirs('shapes-cropped')

    if RECTIFY_SHAPES:
        if not os.path.exists('shapes-rectified'):
            os.makedirs('shapes-rectified')

    if DOWNLOAD_BSDF_BLOBS:
        if not os.path.exists('shapes-bsdf-blobs'):
            os.makedirs('shapes-bsdf-blobs')

    print('processing shapes')
    with open('shapes.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow((
            'shape_id',
            'photo_id',
            'substance_name',
            'name_name',
            'albedo_r',
            'albedo_g',
            'albedo_b',
            'albedo_score',
            'gloss_c',
            'gloss_d',
            'gloss_score',
            'planar',
            'planar_score',
            'normal_matrix',
            'normal_score',
        ))

        shapes = load_shapes()
        photos = {p['pk']: p for p in load_photos()}
        for i, p in enumerate(shapes):
            bsdf = bsdfs.get(p['bsdf_wd'], None)
            if bsdf and bsdf['color_correct'] and bsdf['gloss_correct']:
                (albedo_r, albedo_g, albedo_b), _ = parse_bsdf_albedo(bsdf)
                (gloss_c, gloss_d) = parse_bsdf_c_d(bsdf)
                albedo_score = bsdf['color_correct_score']
                gloss_score = bsdf['gloss_correct_score']
            else:
                bsdf = None
                albedo_r = None
                albedo_g = None
                albedo_b = None
                albedo_score = None
                gloss_c = None
                gloss_d = None
                gloss_score = None

            normal = normals.get(p['rectified_normal'], None)
            if normal and normal['correct'] and p['planar']:
                normal_uvnb = normal['uvnb']
                normal_score = normal['correct_score']
            else:
                normal = None
                normal_uvnb = None
                normal_score = None

            writer.writerow((
                p['pk'],
                p['photo'],
                substances.get(p['substance'], None),
                names.get(p['name'], None),
                albedo_r,
                albedo_g,
                albedo_b,
                albedo_score,
                gloss_c,
                gloss_d,
                gloss_score,
                p['planar'],
                p['planar_score'],
                normal_uvnb,
                normal_score,
            ))

            progress_str = '%s/%s' % (i, len(shapes))

            photo = photos.get(p['photo'], None)
            if photo and CROP_SHAPES:
                pool.apply_async(crop_shape, (p, progress_str))

            if photo and normal and RECTIFY_SHAPES:
                pool.apply_async(rectify_shape, (p, photo, normal, progress_str))

            if bsdf and DOWNLOAD_BSDF_BLOBS:
                filename = os.path.join('shapes-bsdf-blobs', '%s.jpg' % p['pk'])
                url = '%s/%s' % (OPENSURFACES_STATIC_URL_ROOT, bsdf['image_blob'])
                pool.apply_async(download_image, (progress_str, filename, url))


def crop_shape(shape, progress_str):
    """ Crop out the shape's polygon from its photo """

    filename = os.path.join('shapes-cropped', '%s.png' % shape['pk'])
    if os.path.exists(filename):
        print("%s: %s exists: skipping" % (progress_str, filename))
        return

    try:
        photo_image = Image.open('photos/%s.jpg' % shape['photo'])
    except IOError as e:
        print(e)
        return None

    masked, _ = mask_complex_polygon(photo_image, shape['vertices'], shape['triangles'])
    if masked:
        masked.save(filename)
        print("%s: cropped %s" % (progress_str, filename))


def rectify_shape(shape, photo, normal, progress_str):
    """ Rectify the shape (recover its 2D planar image) """

    filename = os.path.join('shapes-rectified', '%s.png' % shape['pk'])
    if os.path.exists(filename):
        print("%s: %s exists: skipping" % (progress_str, filename))
        return

    masked, _ = rectify_shape_from_uvnb(shape, photo, normal['uvnb'])
    if masked:
        masked.save(filename)
        print("%s: rectified %s" % (progress_str, filename))


def render_photo_labels(pool):
    """ Render labels for all photos """

    print("load/sort substances and names from JSON...")
    substances = [(p['pk'], p['name']) for p in load_json('shapes.shapesubstance.json') if not p['fail']]
    names = [(p['pk'],  p['name']) for p in load_json('shapes.shapename.json') if not p['fail']]
    scenes = [(p['pk'],  p['name']) for p in load_json('photos.photoscenecategory.json')]
    substances.sort(key=lambda x: x[1])
    names.sort(key=lambda x: x[1])
    scenes.sort(key=lambda x: x[1])
    assert len(substances) < 254 and len(names) < 254 and len(scenes) < 254

    print("prepare substance color IDs...")
    substance_to_red = {}
    red_color = 1
    with open('label-substance-colors.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(('substance_id', 'substance_name', 'red_color'))
        for pk, name in substances:
            writer.writerow((pk, name, red_color))
            substance_to_red[pk] = red_color
            red_color += 1
    assert red_color <= 255

    print("prepare name color IDs...")
    name_to_green = {}
    green_color = 1
    with open('label-name-colors.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(('name_id', 'name_name', 'green_color'))
        for pk, name in names:
            writer.writerow((pk, name, green_color))
            name_to_green[pk] = green_color
            green_color += 1
    assert green_color <= 255

    print("prepare scene color IDs...")
    scene_to_blue = {}
    blue_color = 1
    with open('label-scene-colors.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(('scene_category_id', 'scene_category_name', 'blue_color'))
        for pk, name in scenes:
            writer.writerow((pk, name, blue_color))
            scene_to_blue[pk] = blue_color
            blue_color += 1
    assert blue_color <= 255

    print("load shapes and photos...")
    photos = load_photos()

    print("map shapes onto photos...")
    photo_shapes = {p['pk']: [] for p in photos}
    for s in load_shapes():
        l = photo_shapes.get(s['photo'], None)
        if l is not None:
            l.append(s)

    print('create directory')
    if not os.path.exists('photos-labels'):
        os.makedirs('photos-labels')

    print("render photos...")
    for i, photo in enumerate(photos):
        progress_str = "%s/%s" % (i, len(photos))
        pool.apply_async(
            render_single_photo_labels,
            (photo, photo_shapes[photo['pk']],
             substance_to_red, name_to_green, scene_to_blue,
             progress_str)
        )


def render_single_photo_labels(photo, photo_shapes, substance_to_red, name_to_green, scene_to_blue, progress_str):
    """ Render labels for one photo """

    filename = os.path.join('photos-labels', '%s.png' % photo['pk'])
    if os.path.exists(filename):
        print("%s: %s exists: skipping" % (progress_str, filename))
        return

    w = int(PHOTO_LABELS_HEIGHT * photo['aspect_ratio'])
    h = int(PHOTO_LABELS_HEIGHT)

    blue_color = scene_to_blue.get(photo['scene_category'], 0)
    labels_image = Image.new(mode='RGB', size=(w, h), color=(0, 0, blue_color))
    draw = ImageDraw.Draw(labels_image)

    # render shapes that are part of this photo
    for shape in photo_shapes:
        # red encodes substance (material)
        # green encodes name (object)
        color = (
            substance_to_red.get(shape['substance'], 0),
            name_to_green.get(shape['name'], 0),
            blue_color
        )

        # extract triangles
        triangles = parse_triangles(shape['triangles'])

        # extract vertices and rescale to pixel coordinates
        vertices = parse_vertices(shape['vertices'])
        vertices = [(int(x * w), int(y * h)) for (x, y) in vertices]

        # render triangles
        for tri in triangles:
            points = [vertices[tri[t]] for t in (0, 1, 2)]
            draw.polygon(points, fill=color)

    del draw
    labels_image.save(filename)
    print('%s: generated %s' % (progress_str, filename))


def download_image(progress_str, filename, url):
    """ Download one image """

    if os.path.exists(filename):
        print('%s: already downloaded %s' % (progress_str, filename))
    else:
        try:
            print('%s: downloading %s --> %s...' % (progress_str, url, filename))
            data = urllib.request.urlopen(url).read()
            with open(filename, 'wb') as f:
                f.write(data)
        except Exception as e:
            print(e)
            os.remove(filename)


def load_json(filename):
    """ Load a JSON object """

    if not os.path.isfile(filename):
        filename = os.path.join('opensurfaces', filename)
    print('parsing %s...' % filename)
    items = json.loads(open(filename).read())
    return [dict(list(p['fields'].items()) + [('pk', p['pk'])]) for p in items]


def load_photos():
    """ Load all photos """

    print('load photos...')
    photos = load_json('photos.photo.json')

    print('throw out photos with incorrect scene category')
    photos = [x for x in photos if x['scene_category_correct']]

    print('throw out synthetic photos')
    photos = [x for x in photos if not x['special']]

    print('sort by num_vertices, then scene_category_correct, then scene_category_correct_score')
    photos.sort(key=lambda x: (x['num_vertices'], x['scene_category_correct'], x['scene_category_correct_score']), reverse=True)

    return photos


def load_shapes():
    """ Load all material shapes """

    print('load shapes...')
    shapes = load_json('shapes.materialshape.json')

    print('throw out synthetic shapes')
    shapes = [x for x in shapes if not x['special']]

    print('throw out low quality shapes')
    shapes = [x for x in shapes if x['high_quality']]

    print('sort by num_vertices')
    shapes.sort(key=lambda x: (x['num_vertices']), reverse=True)

    return shapes


def bbox_vertices(vertices):
    """
    Return bounding box of this object, i.e. ``(min x, min y, max x, max y)``

    :param vertices: List ``[[x1, y1], [x2, y2]]`` or string
        ``"x1,y1,x2,y2,...,xn,yn"``
    """

    if isinstance(vertices, str):
        vertices = parse_vertices(vertices)
    x, y = list(zip(*vertices))  # convert to two lists
    return (min(x), min(y), max(x), max(y))


def parse_vertices(vertices_str):
    """
    Parse vertices stored as a string.

    :param vertices: "x1,y1,x2,y2,...,xn,yn"
    :param return: [(x1,y1), (x1, y2), ... (xn, yn)]
    """
    s = [float(t) for t in vertices_str.split(',')]
    return list(zip(s[::2], s[1::2]))


def parse_triangles(triangles_str):
    """
    Parse a list of vertices.

    :param vertices: "v1,v2,v3,..."
    :return: [(v1,v2,v3), (v4, v5, v6), ... ]
    """
    s = [int(t) for t in triangles_str.split(',')]
    return list(zip(s[::3], s[1::3], s[2::3]))


def parse_bsdf_c_d(bsdf):
    """ Return the (c, d) Ward BRDF coefficients """
    c = bsdf['contrast']
    if c > 1e-3:
        doi = bsdf['doi']
    else:
        # if contrast is 0, then DOI is not defined, so clamp it to 0
        doi = 0
    d = 1 - (0.001 + (15 - doi) * 0.2 / 15)
    return (c, d)


def parse_bsdf_albedo(bsdf):
    """ Return (diffuse RGB albedo, specular RGB albedo) for a BRDF """
    from colormath.color_objects import sRGBColor as RGBColor

    rgb = RGBColor.new_from_rgb_hex(bsdf['color'])
    v = max(rgb.rgb_r, rgb.rgb_b, rgb.rgb_g) / 255.0
    # approximate cielab_inverse_f.
    # we have V instead of L, so the same inverse formula doesn't
    # apply anyway.
    finv = v ** 3
    if bsdf['metallic']:
        rho_s = finv
        s = rho_s / (v * 255.0) if v > 0 else 0
        return (
            (0, 0, 0),
            (s * rgb.rgb_r, s * rgb.rgb_g, s * rgb.rgb_b),
        )
    else:
        rho_d = finv
        t = bsdf['contrast'] + (rho_d * 0.5) ** (1.0 / 3.0)
        rho_s = t ** 3 - rho_d * 0.5
        rho_t = rho_s + rho_d
        if rho_t > 1:
            rho_s /= rho_t
            rho_d /= rho_t
        s = rho_d / (v * 255.0) if v > 0 else 0
        return (
            (s * rgb.rgb_r, s * rgb.rgb_g, s * rgb.rgb_b),
            (rho_s, rho_s, rho_s)
        )


def projection_function(homography):
    """
    Returns a function that applies a homography (3x3 matrix) to 2D tuples
    """
    H = np.copy(homography)

    def project(uv):
        xy = H * np.matrix([[uv[0]], [uv[1]], [1]])
        return (float(xy[0] / xy[2]), float(xy[1] / xy[2]))
    return project


def rectify_shape_from_uvnb(shape, photo, uvnb_json, max_dim=None):
    """
    Rectifies a shape from a photo using the normal matrix (uvnb).
    """

    # Coordinates used in this function:
    #   pq: original pixel coordinates with y down
    #   xy: centered pixel coordinates with y up
    #   uv: in-plane coordinates (arbitrary) with y up
    #   st: rescaled and shifted plane coordinates (fits inside [0,1]x[0,1] but
    #       with correct aspect ratio) with y down
    #   ij: scaled final pixel plane coordinates with y down

    # helper function that applies a homography
    def transform(H, points):
        proj = projection_function(H)
        return [proj(p) for p in points]

    # load original photo info
    try:
        photo_image = Image.open('photos/%s.jpg' % photo['pk'])
    except IOError as e:
        print(e)
        return None
    w, h = photo_image.size
    focal_pixels = 0.5 * max(w, h) / math.tan(math.radians(0.5 * photo['fov']))

    # uvnb: [u v n b] matrix arranged in column-major order
    uvnb = [float(f) for f in json.loads(uvnb_json)]

    # mapping from plane coords to image plane
    M_uv_to_xy = np.matrix([
        [focal_pixels, 0, 0],
        [0, focal_pixels, 0],
        [0, 0, -1]
    ]) * np.matrix([
        [uvnb[0], uvnb[4], uvnb[12]],
        [uvnb[1], uvnb[5], uvnb[13]],
        [uvnb[2], uvnb[6], uvnb[14]]
    ])
    M_xy_to_uv = np.linalg.inv(M_uv_to_xy)

    M_pq_to_xy = np.matrix([
        [1, 0, -0.5 * w],
        [0, -1, 0.5 * h],
        [0, 0, 1],
    ])

    verts_pq = [(v[0] * w, v[1] * h) for v in parse_vertices(shape['vertices'])]
    #print 'verts_pq:', verts_pq

    # estimate rough resolution from original bbox
    if not max_dim:
        min_p, min_q, max_p, max_q = bbox_vertices(verts_pq)
        max_dim = max(max_p - min_p, max_q - min_q)
    #print 'max_dim:', max_dim

    # transform
    verts_xy = transform(M_pq_to_xy, verts_pq)
    #print 'verts_xy:', verts_pq
    verts_uv = transform(M_xy_to_uv, verts_xy)
    #print 'verts_uv:', verts_uv

    # compute bbox in uv plane
    min_u, min_v, max_u, max_v = bbox_vertices(verts_uv)
    max_uv_range = float(max(max_u - min_u, max_v - min_v))
    #print 'max_uv_range:', max_uv_range

    # scale so that st fits inside [0, 1] x [0, 1]
    # (but with the correct aspect ratio)
    M_uv_to_st = np.matrix([
        [1, 0, -min_u],
        [0, -1, max_v],
        [0, 0, max_uv_range]
    ])

    verts_st = transform(M_uv_to_st, verts_uv)
    #print 'verts_st:', verts_st

    M_st_to_ij = np.matrix([
        [max_dim, 0, 0],
        [0, max_dim, 0],
        [0, 0, 1]
    ])

    verts_ij = transform(M_st_to_ij, verts_st)
    #print 'verts_ij:', verts_ij

    # find final bbox
    min_i, min_j, max_i, max_j = bbox_vertices(verts_ij)
    size = (int(math.ceil(max_i)), int(math.ceil(max_j)))
    #print 'i: %s to %s, j: %s to %s' % (min_i, max_i, min_j, max_j)
    #print 'size:', size

    # homography for final pixels to original pixels (ij --> pq)
    M_pq_to_ij = M_st_to_ij * M_uv_to_st * M_xy_to_uv * M_pq_to_xy
    M_ij_to_pq = np.linalg.inv(M_pq_to_ij)
    M_ij_to_pq /= M_ij_to_pq[2, 2]  # NORMALIZE!
    data = M_ij_to_pq.ravel().tolist()[0]
    rectified = photo_image.transform(
        size=size, method=Image.PERSPECTIVE,
        data=data, resample=Image.BICUBIC)

    # crop to polygon
    verts_ij_normalized = [(v[0] / size[0], v[1] / size[1]) for v in verts_ij]
    return mask_complex_polygon(rectified, verts_ij_normalized, shape['triangles'])


def mask_complex_polygon(image, vertices, triangles, bbox_only=False):
    """
    Crops out a complex polygon from an image.  The returned image is cropped
    to the bounding box of the vertices and expanded with transparent pixels to
    a square.

    :param image: path or :class:`PIL.Image`

    :param vertices: List ``[[x1, y1], [x2, y2]]`` or string
        ``"x1,y1,x2,y2,...,xn,yn"``

    :param triangles: List ``[[v1, v2, v3], [v1, v2, v3]]`` or string
        ``"v1,v2,v3,v1,v2,v3,..."``, where ``vx`` is an index into the vertices
        list.

    :param bbox_only: if True, then only the bbox image is returned

    :return: a tuple (masked PIL image, bbox crop PIL image), or None if the bbox is invalid
    """

    if not image:
        return

    if isinstance(vertices, str):
        vertices = parse_vertices(vertices)
    if isinstance(triangles, str):
        triangles = parse_triangles(triangles)
    if isinstance(image, str):
        try:
            image = Image.open(image)
        except IOError as e:
            print(e)
            return None

    # scale up to size
    w, h = image.size
    vertices = [(int(x * w), int(y * h)) for (x, y) in vertices]

    bbox = bbox_vertices(vertices)
    if (len(vertices) < 3 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]):
        return None

    # crop and shift vertices
    image = image.crop(bbox)
    if bbox_only:
        return image

    vertices = [(x - bbox[0], y - bbox[1]) for (x, y) in vertices]

    # draw triangles
    poly = Image.new(mode='RGBA', size=image.size, color=(255, 255, 255, 0))
    draw = ImageDraw.Draw(poly)
    for tri in triangles:
        points = [vertices[tri[t]] for t in (0, 1, 2)]
        draw.polygon(
            points, fill=(255, 255, 255, 255), outline=(255, 255, 255, 255))
    del draw

    # paste into return value image
    ret = Image.new(mode='RGBA', size=image.size, color=(255, 255, 255, 0))
    ret.paste(image, (0, 0), mask=poly)
    return ret, image


def call_with_multiprocessing_pool(func):
    n_cpus = multiprocessing.cpu_count()
    print("multiprocessing: using %s processes" % n_cpus)
    pool = multiprocessing.Pool(n_cpus)
    func(pool)
    pool.close()
    pool.join()


if __name__ == "__main__":

    # check options
    if CROP_SHAPES and not DOWNLOAD_PHOTO_IMAGES:
        print("Error: CROP_SHAPES=True but DOWNLOAD_PHOTO_IMAGES=False")
        sys.exit(1)
    elif RECTIFY_SHAPES and not DOWNLOAD_PHOTO_IMAGES:
        print("Error: RECTIFY_SHAPES=True but DOWNLOAD_PHOTO_IMAGES=False")
        sys.exit(1)

    call_with_multiprocessing_pool(process_photos)
    if RENDER_PHOTO_LABELS:
        call_with_multiprocessing_pool(render_photo_labels)
    call_with_multiprocessing_pool(process_shapes)
