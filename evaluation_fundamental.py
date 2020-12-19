"""
Script for evaluation
This is the evaluation script for image denoising project.

Author: You-Yi Jau, Yiqian Wang
Date: 2020/03/30
"""

import matplotlib
matplotlib.use('Agg') # solve error of tk

import numpy as np
from evaluations.descriptor_evaluation import compute_homography
from evaluations.detector_evaluation import fundamental_repeatability, getFundamentalInliers
import cv2
import matplotlib.pyplot as plt

import logging
import os
from tqdm import tqdm
from utils.draw import plot_imgs

def draw_matches_cv(data, matches, plot_points=True):
    if plot_points:
        keypoints1 = [cv2.KeyPoint(p[1], p[0], 1) for p in data['keypoints1']]
        keypoints2 = [cv2.KeyPoint(p[1], p[0], 1) for p in data['keypoints2']]
    else:
        matches_pts = data['matches']
        keypoints1 = [cv2.KeyPoint(p[0], p[1], 1) for p in matches_pts]
        keypoints2 = [cv2.KeyPoint(p[2], p[3], 1) for p in matches_pts]
        print(f"matches_pts: {matches_pts}")
        # keypoints1, keypoints2 = [], []

    return cv2.drawMatches(data['image1'], keypoints1, data['image2'], keypoints2, matches,
                           None, matchColor=(0,255,0), singlePointColor=(0, 0, 255))

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def find_files_with_ext(directory, extension='.npz', if_int=True):
    # print(os.listdir(directory))
    list_of_files = []
    import os
    if extension == ".npz":
        for l in os.listdir(directory):
            if l.endswith(extension):
                list_of_files.append(l)
                # print(l)
    if if_int:
        list_of_files = [e for e in list_of_files if isfloat(e[:-4])]
    return list_of_files


def to3dim(img):
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    return img

def evaluate(args, **options):
    path = args.path
    files = find_files_with_ext(path)
    correctness = []
    repeatability = []
    mscore = []
    localization_err = []
    rep_thd = 1e-3
    save_file = path + "/result_ext_" + str(rep_thd) + ".txt"
    verbose = True
    top_K = 300
    print("top_K: ", top_K)

    reproduce = True
    if reproduce:
        logging.info("reproduce = True")
        np.random.seed(0)
        print(f"test random # : np({np.random.rand(1)})")

    # create output dir
    if args.outputImg:
        path_warp = path + '/warping'
        os.makedirs(path_warp, exist_ok=True)
        path_match = path + '/matching'
        os.makedirs(path_match, exist_ok=True)
        path_rep = path + '/repeatability' + str(rep_thd)
        os.makedirs(path_rep, exist_ok=True)

    print(f"file: {files[0]}")
    files.sort(key=lambda x: int(x[:-4]))
    from utils.draw import draw_keypoints

    for f in tqdm(files):
        f_num = f[:-4]
        data = np.load(path + '/' + f)
        print("load successfully. ", f)

        real_F = data['fundamental']
        image = data['image']
        warped_image = data['warped_image']
        keypoints = data['prob'][:, [1, 0]]
        print("keypoints: ", keypoints[:3,:])
        warped_keypoints = data['warped_prob'][:, [1, 0]]
        print("warped_keypoints: ", warped_keypoints[:3,:])
        # print("Unwrap successfully.")

        # Repeatability and Localization Error
        if args.repeatability:
            rep, local_err = fundamental_repeatability(data, distance_thresh=rep_thd, verbose=False)
            repeatability.append(rep)
            print("repeatability: %.2f"%(rep))

            if local_err > 0:
                localization_err.append(local_err)
                print('local_err: ', local_err)

            if args.outputImg:
                # img = to3dim(image)
                img = image
                pts = data['prob']
                img1 = draw_keypoints(img*255, pts.transpose())

                # img = to3dim(warped_image)
                img = warped_image
                pts = data['warped_prob']
                img2 = draw_keypoints(img*255, pts.transpose())

                plot_imgs([img1[:,:,[2,1,0]].astype(np.uint8), img2[:,:,[2,1,0]].astype(np.uint8)], titles=['img1', 'img2'], dpi=200)
                plt.title("rep: " + str(repeatability[-1]))
                plt.tight_layout()
                
                plt.savefig(path_rep + '/' + f_num + '.png', dpi=300, bbox_inches='tight')

        # Find matches
        desc = data['desc']
        warped_desc = data['warped_desc']

        # Match the keypoints with the warped_keypoints with nearest neighbor search
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        print("desc: ", desc.shape)
        print("w desc: ", warped_desc.shape)
        cv2_matches = bf.match(desc, warped_desc)
        matches_idx = np.array([m.queryIdx for m in cv2_matches])
        m_keypoints = keypoints[matches_idx, :]
        matches_idx = np.array([m.trainIdx for m in cv2_matches])
        m_dist = np.array([m.distance for m in cv2_matches])
        m_warped_keypoints = warped_keypoints[matches_idx, :]
        matches = np.hstack((m_keypoints[:, [1, 0]], m_warped_keypoints[:, [1, 0]]))
        print(f"matches: {matches.shape}")

        # Compute matching score
        def warpLabels(pnts, homography, H, W):
            import torch
            """
            input:
                pnts: numpy
                homography: numpy
            output:
                warped_pnts: numpy
            """
            from utils.utils import warp_points
            from utils.utils import filter_points
            pnts = torch.tensor(pnts).long()
            homography = torch.tensor(homography, dtype=torch.float32)
            warped_pnts = warp_points(torch.stack((pnts[:, 0], pnts[:, 1]), dim=1),
                                        homography)  # check the (x, y)
            warped_pnts = filter_points(warped_pnts, torch.tensor([W, H])).round().long()
            return warped_pnts.numpy()

        H, W, d = image.shape
        # Matching score should be symmetrically computed and averaged
        # TODO: Matching score definition should be more precisely seared.
        result = {}
        inliers = getFundamentalInliers(matches, real_F, epi=rep_thd)
        result['inliers_bf'] = inliers
        score = inliers.sum() / (keypoints.shape[0] + warped_keypoints.shape[0] - inliers.sum())

        inliers = getFundamentalInliers(np.concatenate((matches[:,2:4], matches[:,:2]), axis=1), real_F.T, epi=rep_thd)
        result['inliers_bf'] += inliers
        score += inliers.sum() / (keypoints.shape[0] + warped_keypoints.shape[0] - inliers.sum())
        score /= 2

        print("m. score: ", score)
        mscore.append(score)

        homography_thresh = [1,3,5,10,20,50]
        if args.homography:
            # estimate result
            ##### check
            #####
            result = compute_homography(data, correctness_thresh=homography_thresh)
            correctness.append(result['correctness'])

            if args.outputImg:
                # draw warping
                output = result

                img1 = image
                img2 = warped_image
                H = output['homography']

                ## plot filtered image
                warped_img1 = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))
                plot_imgs([img1, img2, warped_img1], titles=['img1', 'img2', 'warped_img1'], dpi=200)
                plt.tight_layout()
                # plt.savefig(path_warp + '/' + f_num + '_fil.png')
                plt.savefig(path_warp + '/' + f_num + '.png')

                # plt.show()

                # draw matches
                result['image1'] = image
                result['image2'] = warped_image
                matches = np.array(result['cv2_matches'])
                # ratio = 0.2
                # ran_idx = np.random.choice(matches.shape[0], int(matches.shape[0]*ratio))

                # img = draw_matches_cv(result, matches[ran_idx], plot_points=True)
                img = draw_matches_cv(result, matches, plot_points=True)
                # filename = "correspondence_visualization"
                plot_imgs([img], titles=["Two images feature correspondences"], dpi=200)
                plt.tight_layout()
                plt.savefig(path_match + '/' + f_num + 'cv.png', bbox_inches='tight')
                plt.close('all')
                # pltImshow(img)

        if args.plotMatching:
            matches = result['matches'] # np [N x 4]
            if matches.shape[0] > 0:
                from utils.draw import draw_matches
                filename = path_match + '/' + f_num + 'm.png'
                # ratio = 0.1
                inliers = result['inliers_bf']

                matches_in = matches[inliers == True]
                matches_out = matches[inliers == False]

                image = data['image']
                warped_image = data['warped_image']
                ## outliers
                # matches_temp, _ = get_random_m(matches_out, ratio)
                # print(f"matches_in: {matches_in.shape}, matches_temp: {matches_temp.shape}")
                draw_matches(image, warped_image, matches_out, lw=0.5, color='r',
                            filename=None, show=False, if_fig=True)
                ## inliers
                # matches_temp, _ = get_random_m(matches_in, ratio)
                draw_matches(image, warped_image, matches_in, lw=1.0, 
                        filename=filename, show=False, if_fig=False)

    if args.repeatability:
        repeatability_ave = np.array(repeatability).mean()
        localization_err_m = np.array(localization_err).mean()
        print("repeatability: ", repeatability_ave)
        print("localization error over ", len(localization_err), " images : ", localization_err_m)
    if args.homography:
        correctness_ave = np.array(correctness).mean(axis=0)
        print("homography estimation threshold", homography_thresh)
        print("correctness_ave", correctness_ave)

    mscore_m = np.array(mscore).mean(axis=0)
    print("matching score", mscore_m)
    print("end")

    # save to files
    with open(save_file, "a") as myfile:
        myfile.write("path: " + path + '\n')
        myfile.write("output Images: " + str(args.outputImg) + '\n')
        if args.repeatability:
            myfile.write("repeatability threshold: " + str(rep_thd) + '\n')
            myfile.write("repeatability: " + str(repeatability_ave) + '\n')
            myfile.write("localization error: " + str(localization_err_m) + '\n')
        if args.homography:
            myfile.write("Homography estimation: " + '\n')
            myfile.write("Homography threshold: " + str(homography_thresh) + '\n')
            myfile.write("Average correctness: " + str(correctness_ave) + '\n')
        myfile.write("matching score: " + str(mscore_m) + '\n')

        if verbose:
            myfile.write("====== details =====" + '\n')
            for i in range(len(files)):

                myfile.write("file: " + files[i])
                if args.repeatability:
                    myfile.write("; rep: " + str(repeatability[i]))
                if args.homography:
                    myfile.write("; correct: " + str(correctness[i]))
                # matching
                myfile.write("; mscore: " + str(mscore[i]))
                myfile.write('\n')
            myfile.write("======== end ========" + '\n')

    dict_of_lists = {
        'repeatability': repeatability,
        'localization_err': localization_err,
        'correctness': np.array(correctness),
        'homography_thresh': homography_thresh,
        'mscore': mscore
    }

    filename = f'{save_file[:-4]}.npz'
    logging.info(f"save file: {filename}")
    np.savez(
        filename,
        **dict_of_lists,
    )


if __name__ == '__main__':
    import argparse


    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--sift', action='store_true', help='use sift matches')
    parser.add_argument('-o', '--outputImg', action='store_true')
    parser.add_argument('-r', '--repeatability', action='store_true')
    parser.add_argument('-homo', '--homography', action='store_true')
    parser.add_argument('-plm', '--plotMatching', action='store_true')
    args = parser.parse_args()
    evaluate(args)
