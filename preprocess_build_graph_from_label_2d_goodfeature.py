import numpy as np
import skfmm
from tqdm import tqdm
import os.path as osp
import pickle
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from skimage.draw import line
import time
import cv2
from scipy.spatial import KDTree, cKDTree
from skimage.morphology import skeletonize
from scipy import ndimage as ndi
from skimage import measure
import math
from utils import safe_slice_array, insert_array, define_fd
from utils_preprocess import choose_max_connected_component
from preprocess_build_graph_from_label_2d import uncertainty_from_pseudo_label

def safe_round(coord, shape):
    coord = np.round(coord).astype(np.int32)
    for i in range(len(coord)):
        if coord[i] < 0:
            coord[i] = 0
        if coord[i] >= shape[i]:
            coord[i] = shape[i] - 1
    return coord

def find_endpoints_2d(skel):
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)
    neigh_count = ndi.convolve(skel.astype(np.uint8), kernel, mode='constant', cval=0)
    ys, xs = np.nonzero((skel > 0) & (neigh_count == 1))
    return list(zip(ys, xs))

def downsample_and_dilate(mask, downsample=2, dilation_size=5):
    assert len(mask.shape) == 2
    small_mask = cv2.resize(mask.astype(np.float32), (mask.shape[1] // downsample, mask.shape[0] // downsample), interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    dilated_mask = cv2.dilate(small_mask.astype(np.uint8), kernel)
    dilated_mask = cv2.resize(dilated_mask.astype(np.float32), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    dilated_mask = (dilated_mask > 0.5).astype(np.uint8)
    return dilated_mask

def find_endpoints(skel):
    """Return skeleton endpoint coordinates (row, column) with one 8-connected neighbor."""
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)
    neigh_count = ndi.convolve(skel.astype(np.uint8), kernel, mode='constant', cval=0)
    ys, xs = np.nonzero((skel > 0) & (neigh_count == 1))
    return list(zip(ys, xs))

def local_tangent_at(pt, skel, radius=5):
    """Estimate a local skeleton tangent with PCA; return None when too few points are available."""
    y, x = pt
    h, w = skel.shape
    miny, maxy = (max(0, y - radius), min(h, y + radius + 1))
    minx, maxx = (max(0, x - radius), min(w, x + radius + 1))
    sub = skel[miny:maxy, minx:maxx]
    coords = np.argwhere(sub)
    if coords.shape[0] < 2:
        return None
    coords_global = coords + np.array([miny, minx])
    centered = coords_global - coords_global.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    vec = vt[0]
    norm = np.linalg.norm(vec)
    if norm < 1e-09:
        return None
    return vec / norm

def angle_between(u, v):
    """Return the unsigned angle between two vectors in degrees."""
    dot = float(np.dot(u, v))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))

def connect_skeleton_endpoints(skel, max_dist=40, angle_thresh_deg=45, local_radius=5, allow_multi_connect=False, connect_to_tree_thres=0):
    """Connect nearby endpoints from different components, optionally extending endpoints to another skeleton component. Return the updated mask and connection list."""
    assert skel.ndim == 2, 'Only 2D skeletons are supported'
    sk = skel.astype(bool).copy()
    comp_labels = measure.label(sk, connectivity=2)
    endpoints = find_endpoints(sk)
    if len(endpoints) < 2:
        return (sk, [])
    tangents = {}
    comp_of = {}
    for pt in endpoints:
        tangents[pt] = local_tangent_at(pt, sk, radius=local_radius)
        comp_of[pt] = int(comp_labels[pt])
    connections = []
    used = set()
    for i in range(len(endpoints)):
        p1 = endpoints[i]
        if not allow_multi_connect and p1 in used:
            continue
        t1 = tangents.get(p1)
        if t1 is None:
            continue
        for j in range(i + 1, len(endpoints)):
            p2 = endpoints[j]
            if not allow_multi_connect and p2 in used:
                continue
            if comp_of.get(p1, 0) != 0 and comp_of.get(p2, 0) != 0 and (comp_of[p1] == comp_of[p2]):
                continue
            t2 = tangents.get(p2)
            if t2 is None:
                continue
            vec = np.array(p2) - np.array(p1)
            dist = np.linalg.norm(vec)
            if dist > max_dist:
                continue
            vec_u = vec / (dist + 1e-09)
            ang1 = angle_between(t1, vec_u)
            ang2 = angle_between(t2, -vec_u)
            if ang1 <= angle_thresh_deg and ang2 <= angle_thresh_deg:
                rr, cc = line(p1[0], p1[1], p2[0], p2[1])
                sk[rr, cc] = True
                connections.append((p1, p2))
                if not allow_multi_connect:
                    used.add(p1)
                    used.add(p2)
                break
    if connect_to_tree_thres > 0:
        comps = measure.label(sk, connectivity=2)
        endpoints = find_endpoints_2d(sk)
        if len(endpoints) == 0:
            return (sk, connections)
        sk_coords = np.argwhere(sk)
        comp_ids = comps[sk_coords[:, 0], sk_coords[:, 1]]
        if sk_coords.shape[0] == 0:
            return (sk, connections)
        tree_all = cKDTree(sk_coords)
        candidates_by_comp = {}
        for ep in endpoints:
            comp_ep = comps[ep]
            mask_other = comp_ids != comp_ep
            if not np.any(mask_other):
                continue
            coords_other = sk_coords[mask_other]
            if coords_other.shape[0] == 0:
                continue
            tree_other = cKDTree(coords_other)
            dist, idx = tree_other.query(np.array(ep), k=1)
            if dist <= connect_to_tree_thres:
                nearest_pt = tuple(coords_other[int(idx)])
                lst = candidates_by_comp.setdefault(comp_ep, [])
                lst.append((ep, float(dist), nearest_pt))
        for comp_id, lst in candidates_by_comp.items():
            lst_sorted = sorted(lst, key=lambda x: x[1])
            chosen = lst_sorted[0]
            ep, dist, nearest_pt = chosen
            rr, cc = line(ep[0], ep[1], nearest_pt[0], nearest_pt[1])
            sk[rr, cc] = True
            connections.append((ep, nearest_pt, 'ep-trunk'))
    return (sk, connections)

def find_center_of_block(block, max_conn=True, center_type='weight'):
    assert center_type in ['weight', 'inscribed']
    if np.sum(block) == 0:
        return np.array(block.shape) / 2
    if max_conn:
        block = choose_max_connected_component(block)
    center = np.zeros(2, dtype=np.float32)
    if center_type == 'weight':
        for i in range(2):
            center[i] = np.mean(np.where(block > 0)[i])
    elif center_type == 'inscribed':
        block_dt = distance_transform_edt(block)
        center = np.where(block_dt == np.max(block_dt))
        center = np.array(center)[:, 0]
    center = center + 0.5
    return center

def get_time_map(coord1, mask, narrow=0, max_block_size=64):
    coord1_round = safe_round(coord1, mask.shape)
    phi = np.ones_like(mask, dtype=np.float32) * -1
    phi[coord1_round[0], coord1_round[1]] = 1
    mask = (1 - mask) * 0.01 + mask
    start = coord1_round - max_block_size // 2
    end = coord1_round + max_block_size // 2
    phi_block, true_start, true_end = safe_slice_array(phi, start, end)
    mask_block, _, _ = safe_slice_array(mask, start, end)
    time_map_block = skfmm.travel_time(phi_block, mask_block, narrow=narrow)
    if narrow > 0:
        time_map_block = time_map_block.filled(9999999)
    time_map = np.ones_like(mask) * 9999999
    time_map = insert_array(time_map, time_map_block, true_start)
    return time_map

def determine_uncertainty(uncertainty_mask, coord):
    surrounding_coords_lst = []
    for j in range(2):
        for k in range(2):
            if np.all(np.floor(coord) + np.array([j, k]) < uncertainty_mask.shape):
                surrounding_coords_lst.append(np.floor(coord) + np.array([j, k]))
    surrounding_coords = np.array(surrounding_coords_lst)
    surrounding_coords = surrounding_coords.astype(np.int32)
    distances = np.linalg.norm(surrounding_coords - coord, axis=1)
    uncertainty = np.sum(uncertainty_mask[surrounding_coords[:, 0], surrounding_coords[:, 1]] / (distances + 1e-06)) / np.sum(1 / (distances + 1e-06))
    return uncertainty

def do_find_center_of_block_new(block_pseudo, block_uncertainty, certain_pseudo_only=True, center_type='weight'):
    assert center_type in ['weight', 'inscribed']
    if certain_pseudo_only:
        block_pseudo = block_pseudo * block_uncertainty
    if np.sum(block_pseudo) > 0:
        return find_center_of_block(block_pseudo, max_conn=True, center_type=center_type)
    elif np.sum(1 - block_uncertainty) > 0:
        return find_center_of_block(1 - block_uncertainty, max_conn=True, center_type=center_type)
    else:
        assert np.sum(block_pseudo) == 0
        return find_center_of_block(block_pseudo, max_conn=False, center_type=center_type)

def build_graph_with_uncertainty(label, uncertainty_mask, cell_size=(8, 8), coarse=False, center_type='weight', new_find_center=False, certain_pseudo_only=True, tn_do_not_connect=0, do_determine_if_connect=True, image_mask=None, goodfeatures={'num': 2500, 'thres': 0.0005, 'min_dist': 3}, centerline={'use': False, 'downsample': 2}, connect_to_tree_thres=0, max_edge_for_each_node=999999):
    cell_num = label.shape // np.array(cell_size)
    gt = np.zeros(np.prod(cell_num), dtype=np.int32)
    uncertainty = np.zeros(np.prod(cell_num), dtype=np.float32)
    label_plus_uncertainty = label + (1 - uncertainty_mask)
    coords = np.zeros((np.prod(cell_num), 2))
    for y in range(cell_num[0]):
        for x in range(cell_num[1]):
            one_d_idx = y * cell_num[1] + x
            block = label[y * cell_size[0]:(y + 1) * cell_size[0], x * cell_size[1]:(x + 1) * cell_size[1]]
            block_label_plus_uncertainty = label_plus_uncertainty[y * cell_size[0]:(y + 1) * cell_size[0], x * cell_size[1]:(x + 1) * cell_size[1]]
            block_uncertainty = uncertainty_mask[y * cell_size[0]:(y + 1) * cell_size[0], x * cell_size[1]:(x + 1) * cell_size[1]]
            if new_find_center:
                center = do_find_center_of_block_new(block, block_uncertainty, certain_pseudo_only=certain_pseudo_only, center_type=center_type)
            else:
                center = find_center_of_block(block_label_plus_uncertainty, center_type=center_type)
            center = center + np.array([y, x]) * cell_size
            coords[one_d_idx] = center
            if coarse:
                gt[one_d_idx] = np.sum(block) > np.prod(block.shape) * 0.001
            else:
                gt[one_d_idx] = determine_uncertainty(label, center) > 0.5
            uncertainty[one_d_idx] = determine_uncertainty(uncertainty_mask, center) > 0.5
            if image_mask is not None:
                if determine_uncertainty(image_mask, center) < 0.5:
                    gt[one_d_idx] = 0
                    uncertainty[one_d_idx] = 1
    area_interest = (uncertainty_mask < 0.5) | (label > 0.5)
    area_interest = area_interest.astype(np.float32)
    area_interest_skeleton = skeletonize(area_interest > 0.5).astype(np.float32)
    area_interest_skeleton, _ = connect_skeleton_endpoints(area_interest_skeleton, max_dist=20, angle_thresh_deg=360, local_radius=5, connect_to_tree_thres=connect_to_tree_thres)
    area_interest = downsample_and_dilate(area_interest_skeleton > 0.5, 1, 5) | (area_interest > 0.5)
    area_interest = area_interest.astype(np.float32)
    if goodfeatures is not None:
        corners = cv2.goodFeaturesToTrack(area_interest_skeleton, goodfeatures['num'], goodfeatures['thres'], goodfeatures['min_dist'])
        corners = corners[:, 0, ::-1]
        corners_gt = label[corners.astype(np.int32)[:, 0], corners.astype(np.int32)[:, 1]]
        corners_uncertainty = uncertainty_mask[corners.astype(np.int32)[:, 0], corners.astype(np.int32)[:, 1]]
        coords = np.concatenate([coords, corners], axis=0)
        gt = np.concatenate([gt, (corners_gt > 0.5).astype(np.int32)], axis=0)
        uncertainty = np.concatenate([uncertainty, (corners_uncertainty > 0.5).astype(np.float32)], axis=0)
    if centerline and centerline['use']:
        assert goodfeatures is None
        area_interest_ds = cv2.resize(area_interest.astype(np.float32), (area_interest.shape[1] // centerline['downsample'], area_interest.shape[0] // centerline['downsample']), interpolation=cv2.INTER_NEAREST)
        area_interest_skeleton_ds = skeletonize(area_interest_ds > 0.5).astype(np.float32)
        ys, xs = np.where(area_interest_skeleton_ds > 0.5)
        skeleton_coords = np.stack([ys * centerline['downsample'] + centerline['downsample'] // 2, xs * centerline['downsample'] + centerline['downsample'] // 2], axis=1)
        skeleton_gts = label[skeleton_coords.astype(np.int32)[:, 0], skeleton_coords.astype(np.int32)[:, 1]]
        skeleton_uncertainties = uncertainty_mask[skeleton_coords.astype(np.int32)[:, 0], skeleton_coords.astype(np.int32)[:, 1]]
        skeleton_tn = (skeleton_gts < 0.5) & (skeleton_uncertainties > 0.5)
        skeleton_uncertainties[skeleton_tn] = 0
        coords = np.concatenate([coords, skeleton_coords], axis=0)
        gt = np.concatenate([gt, (skeleton_gts > 0.5).astype(np.int32)], axis=0)
        uncertainty = np.concatenate([uncertainty, (skeleton_uncertainties > 0.5).astype(np.float32)], axis=0)
    tree = KDTree(coords)
    edges_lst = []
    edge_weights_lst = []
    time_thres = cell_size[0] * 2

    def determine_if_connect(uncertainty, label, i, j, tn_do_not_connect=0):
        if tn_do_not_connect == 2:
            if uncertainty[j] == 1 and label[j] == 0 or (uncertainty[i] == 1 and label[i] == 0):
                return False
        elif tn_do_not_connect == 1:
            if (uncertainty[j] == 1 and label[j] == 0) and (uncertainty[i] == 1 and label[i] == 0):
                return False
        else:
            pass
        if label[j] == 1 or label[i] == 1 or uncertainty[j] == 0 or (uncertainty[i] == 0):
            return True
        else:
            return False
    with tqdm(range(len(gt))) as t:
        for i in t:
            if uncertainty[i] == 0 or gt[i] == 1:
                time_map = get_time_map(coords[i], area_interest, narrow=0, max_block_size=cell_size[0] * 3)
                this_edges_lst = []
                this_weights_lst = []
                this_candidate_points_lst = []
                for j in tree.query_ball_point(coords[i], r=cell_size[0] * 1.5):
                    if do_determine_if_connect:
                        flag_connect = determine_if_connect(uncertainty, gt, i, j, tn_do_not_connect=tn_do_not_connect)
                    else:
                        flag_connect = True
                    if flag_connect:
                        coords_round = safe_round(coords[j], label.shape)
                        time = time_map[coords_round[0], coords_round[1]]
                        if time <= time_thres * 2.5:
                            this_candidate_points_lst.append((j, time))
                this_candidate_points_lst = sorted(this_candidate_points_lst, key=lambda x: x[1])
                for idx_cand in range(min(len(this_candidate_points_lst), max_edge_for_each_node)):
                    j, time = this_candidate_points_lst[idx_cand]
                    this_edges_lst.append([i, j])
                    this_weights_lst.append(time_thres / (time + time_thres))
                edges_lst.extend(this_edges_lst)
                this_weights_lst = np.array(this_weights_lst)
                edge_weights_lst.extend(this_weights_lst)
    edges = np.array(edges_lst, dtype=np.int32)
    edge_weights = np.array(edge_weights_lst, dtype=np.float32)
    print(f'Total positive points: {np.sum(gt)}. Total negative points: {np.sum(1 - gt)}. Total edges: {len(edges)}')
    return (gt, uncertainty, coords, edges, edge_weights)

def main():
    import argparse
    import json
    from utils import load_yaml
    from submission_config import dataset_root, graph_folder, cases
    parser = argparse.ArgumentParser(description='Build two-scale vessel graphs from pseudo labels; no manual labels are read.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--splits', nargs='+', default=['train', 'val'], choices=['train', 'val'])
    args = parser.parse_args()
    config = load_yaml(args.config)
    root = dataset_root(config)
    pp = config.get('preprocessing', {})
    connect_distance = pp.get('connect_to_tree_thres', 20)
    version = 2508 if connect_distance > 0 else 1508
    if config['graph_version'] != version or config.get('graph_psuncert') or config.get('graph_trained_num') is not None:
        raise ValueError('This driver supports source pseudo-label graphs v1508 (distance=0) and v2508 (distance>0).')
    sides = [config.get('highres_graph_sidelen', 16), config.get('lowres_graph_sidelen', 32)]
    size = tuple(config.get('image_size', [1024, 1024]))
    for split in args.splits:
        pseudo_dir = root / ('PLPB_mask_train' if split == 'train' else 'PLPB_mask_test')
        for side in sides:
            dst = graph_folder(config, split, side)
            dst.mkdir(parents=True, exist_ok=True)
            for case in cases(config, split):
                with (pseudo_dir / (case.name + '_pseudo.pkl')).open('rb') as stream:
                    pseudo = pickle.load(stream).cpu().numpy()[0, 0]
                with (pseudo_dir / (case.name + '_mask.pkl')).open('rb') as stream:
                    certainty = 1 - pickle.load(stream).cpu().numpy()[0, 0]
                if pseudo.shape != size or certainty.shape != size:
                    raise ValueError(f'{case.name}: pseudo maps must have shape {size}')
                fov = None
                if config.get('graph_masked', False):
                    fov = cv2.imread(str(case / 'mask.png'), cv2.IMREAD_GRAYSCALE)
                    if fov is None or fov.shape != size:
                        raise ValueError(f'Missing or wrong-sized field-of-view mask: {case / "mask.png"}')
                    fov = fov.astype(np.float32) / 255
                values = build_graph_with_uncertainty(
                    pseudo, certainty, cell_size=(side, side), coarse=False,
                    center_type='inscribed', new_find_center=True, certain_pseudo_only=False,
                    tn_do_not_connect=2, do_determine_if_connect=True, image_mask=fov,
                    goodfeatures=None, centerline={'use': True, 'downsample': max(1, side // 8)},
                    connect_to_tree_thres=connect_distance, max_edge_for_each_node=5)
                graph = dict(zip(['gt', 'uncertainty', 'coords', 'edges', 'edge_weights'], values))
                graph['edges'] = graph['edges'].reshape(-1, 2)
                with (dst / (case.name + '.pkl')).open('wb') as stream:
                    pickle.dump(graph, stream)
                print(f'Saved {dst / (case.name + ".pkl")}: {len(graph["coords"])} nodes, {len(graph["edges"])} edges')
            (dst / 'preprocessing.json').write_text(json.dumps({'graph_version': version, 'side': side,
                'connect_to_tree_thres': connect_distance, 'max_edges_per_node': 5}, indent=2))


if __name__ == '__main__':
    main()
