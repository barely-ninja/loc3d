from sys import argv
from json import load, dump
from numpy import sin, cos, hstack, zeros, array, squeeze, multiply, asscalar
from scipy.optimize import minimize
from math import pi, atan

def rot_3d(ang_vec):
    '3d rotation matrix'
    s = sin(ang_vec*pi/180.)
    c = cos(ang_vec*pi/180.)
    rx = array([[1, 0, 0], [0, c[0], -s[0]], [0, s[0], c[0]]])
    ry = array([[c[1], 0, s[1]], [0, 1, 0], [-s[1], 0, c[1]]])
    rz = array([[c[2], -s[2], 0] ,[s[2], c[2], 0], [0, 0, 1]])
    return rz@ry@rx

def extrinsic(ang_vec, offset):
    'rotation after offset in world coords'
    rot_m = rot_3d(ang_vec)
    off_vec = rot_m.dot(-1*offset).reshape(3, 1)
    return hstack((rot_m, off_vec))

def project(vect):
    'Projection transform'
    return vect/vect[2]

def intrinsic(f_pix, x_off, y_off):
    'Pinhole cam model with square pixels'
    return array([
        [f_pix, 0., x_off],
        [0., f_pix, y_off],
        [0., 0., 1.]
    ])

def make_pack_func(**inits):
    'Closure for parameter transforms'
    f_pix = 10.*inits['pic_size'][0]

    num_ref = len(inits['ref_pos'])
    sums = [sum([l['coords'][i] for l in inits['ref_pos']]) for i in range(2)]
    rel_offset = [inits['cam_pos'][i]-sums[i]/num_ref for i in range(2)]
    azimuth = atan(rel_offset[0]/rel_offset[1])*180./pi
    angles = [90., azimuth, 0.]

    offset = inits['cam_pos']

    init_vector = array([f_pix]+angles+offset)
    uncertainty = array([f_pix*0.9, 1e1, 3e1, 3e1, 1e4, 1e4, 1e2])
    def unpack(opt_vector):
        'Unpacks opt vector to original parameter scale'
        unpacked = multiply(opt_vector, uncertainty) + init_vector
        return unpacked[0], unpacked[1:4], unpacked[4:]
    return unpack


def make_residual_func(**inits):
    'Closure for residual function'
    unpack = make_pack_func(**inits)
    def res_func(x):
        'Calculates sum of squared residuals'
        nonlocal inits, unpack
        f_pix, angle, offset = unpack(x)
        intr = intrinsic(f_pix, inits['pic_size'][0]/2, inits['pic_size'][1]/2)
        extr = extrinsic(angle, offset)
        err = 0
        for known in inits['ref_pos']:
            known_coords = array(known['coords']+[1.]).reshape(4, 1)
            img_coords = intr@(project(extr@known_coords))
            diff = img_coords - array(known['pic_xy']+[1.]).reshape(3, 1)
            err += asscalar(diff.T@diff)
        print(x, err)
        return err
    return unpack, res_func

def estim_all(**inits):
    'Finds pinhole camera transform parameters'
    unpack, res_func = make_residual_func(**inits)
    x_init = zeros(7)
    bounds = [(-1, 1)]*7
    result = minimize(res_func, x_init,
                      method="L-BFGS-B", bounds=bounds, options={'eps':1e-5})
    return unpack(result.x)

def main(params):
    'I/O wrapper'
    try:
        config_fn = params[1]
    except KeyError:
        print('Please specify config file name as first argument')

    with open(config_fn, 'rt') as cfg_file:
        cfg = load(cfg_file)

    params = estim_all(
        pic_size=cfg['pic_size'],
        cam_pos=cfg['camera_guess'],
        ref_pos=cfg['known']
    )

    with open(cfg['cal_file'], 'wt') as output_file:
        output = {
            'pic_center': [cfg['pic_size'][i]/2 for i in range(2)],
            'focal_length_pix': params[0],
            'camera_rotation_angles': [x for x in params[1]],
            'camera_coordinates': [x for x in params[2]],

        }
        dump(output, output_file)

if __name__ == '__main__':
    main(argv)
