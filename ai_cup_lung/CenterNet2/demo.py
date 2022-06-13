
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
 
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
 
from predictor import VisualizationDemo
from centernet.config import add_centernet_config
# constants
WINDOW_NAME = "CenterNet2 detections"
 
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
NUM_CLASSES=1
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cuda'

    add_centernet_config(cfg)
 
    cfg.MODEL.CENTERNET.NUM_CLASSES=NUM_CLASSES
    cfg.MODEL.ROI_HEADS.NUM_CLASSES =NUM_CLASSES
 
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    if cfg.MODEL.META_ARCHITECTURE in ['ProposalNetwork', 'CenterNetDetector']:
        cfg.MODEL.CENTERNET.INFERENCE_TH = args.confidence_threshold
        cfg.MODEL.CENTERNET.NMS_TH = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg
 
 
def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/CenterNet2_R2-101-DCN_896_4x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    #D:/Competition/lung/OBJ/Test_Images/
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", default="D:/Competition/lung/OBJ/private/",nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        default="CenterNet2/test_result",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', 'weight/best_model_res2net101_FPN_896.pth'],
        nargs=argparse.REMAINDER,
    )
    return parser
 
 
if __name__ == "__main__":
    json_path = "D:/JIM/CenterNet2-master/output/txt_box/predict.json"

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    print(demo)
    output_file = None
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            files = os.listdir(args.input[0])
            args.input = [args.input[0] + x for x in files]
            assert args.input, "The input path(s) was not found"
        visualizer = VideoVisualizer(
            MetadataCatalog.get(
                cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
            ), 
            instance_mode=ColorMode.IMAGE)
        all_img = os.listdir(args.input)

        dic = dict.fromkeys(all_img,0)
        txtpath = os.path.join(json_path)
        all_box = 0
        for path in tqdm.tqdm(os.listdir(args.input)):
            args = get_parser().parse_args()
            cfg = setup_cfg(args)
            demo = VisualizationDemo(cfg)
            cfthresh = 0.1
            path = os.path.join(args.input, path)
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            visualizer.metadata.thing_classes[:10]=["stas"]

            predictions, visualized_output = demo.run_on_image(
                img, visualizer=visualizer)


            new_name = path.replace('D:/Competition/lung/OBJ/Test_Images/', '')
            box_ma = []
            f = open(txtpath, 'w')
            for x in range(len(predictions["instances"])): 
                # if predictions["instances"].pred_classes[x].item() == 0:
                    aa = [round(predictions["instances"].pred_boxes.tensor[x,0].item()), 
                        round(predictions["instances"].pred_boxes.tensor[x,1].item()),
                        round(predictions["instances"].pred_boxes.tensor[x,2].item()),
                        round(predictions["instances"].pred_boxes.tensor[x,3].item()),
                        round(predictions["instances"].scores[x].item(), 5)]
                    all_box=all_box+1
                    box_ma.append(aa)
            dic[new_name] = box_ma
            print("box_num",all_box)

            if 'instances' in predictions:
                logger.info(
                    "{}: detected {} instances in {:.2f}s".format(
                        path, len(predictions["instances"]), time.time() - start_time
                    )
                )
            else:
                logger.info(
                    "{}: detected {} instances in {:.2f}s".format(
                        path, len(predictions["proposals"]), time.time() - start_time
                    )
                )
 
            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                    visualized_output.save(out_filename)
                else:
                    # assert len(args.input) == 1, "Please specify a directory with args.output"
                    # out_filename = args.output
                    if output_file is None:
                        width = visualized_output.get_image().shape[1]
                        height = visualized_output.get_image().shape[0]
                        frames_per_second = 15
                        output_file = cv2.VideoWriter(
                            filename=args.output,
                            # some installation of opencv may not support x264 (due to its license),
                            # you can try other format (e.g. MPEG)
                            fourcc=cv2.VideoWriter_fourcc(*"x264"),
                            fps=float(frames_per_second),
                            frameSize=(width, height),
                            isColor=True,
                        )
                    lastimg = output_file.write(visualized_output.get_image()[:, :, ::-1])
                    print(lastimg)
                    cv2.imshow("", visualized_output.get_image()[:, :, ::-1])
                    cv2.waitKey(0)

            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(1 ) == 27:
                    break  # esc to quit
        dica = str(dic).replace("'","\"")
        f.write(dica)     
        f.close()
