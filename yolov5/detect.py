import argparse
from tqdm import tqdm

from utils import google_utils
from utils.datasets import *
from utils.utils import *


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default=None, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--demo', action='store_true', help='produce a video & images for demo')

    parser.add_argument('--weights', type=str, default='weights/yolov5l.pt', help='model.pt path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()

    assert opt.video, 'Error: wrong video path: %s' % opt.video
    assert opt.weights, 'Error: wrong model weight path: %s ' % opt.weights
    opt.img_size = check_img_size(opt.img_size)
    return opt


def obj_detect(opt):
    vid_path, weights, imgsz = opt.video, opt.weights, opt.img_size

    # Initialize
    device = torch_utils.select_device(opt.device)
    half = False  # half precision only supported on CUDA

    # Load model
    google_utils.attempt_download(weights)
    # Cause issue when running out of yolo folder !!!
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    dataset = LoadImages(vid_path, img_size=imgsz)

    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    all_bbox = []
    all_score = []
    pbar = tqdm(total=0)  # manual visualize
    pbar.total = dataset.nframes
    if opt.demo:
        vid_writer = cv2.VideoWriter(os.path.splitext(vid_path)[0] + '_obj.mp4', cv2.VideoWriter_fourcc(*opt.fourcc),
                                     dataset.fps, (dataset.width, dataset.height))
        out_dir = os.path.splitext(vid_path)[0] + '_obj'
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

    for path, img, im0, vid_cap in dataset:
        pbar.n = dataset.frame
        pbar.refresh()

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                xyxy_npy = det.detach().to('cpu').numpy()
                xyxy_person = xyxy_npy[xyxy_npy[:,-1]==0]
                all_bbox.append(xyxy_person[:, :-2])
                all_score.append(xyxy_person[:, -2])
                # Write results
                for *xyxy, conf, cls in det:
                    if cls != 0:
                        continue
                    
                    if opt.demo:
                        label_plot = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label_plot, color=colors[int(cls)], line_thickness=3)
            else:
                all_bbox.append(np.array([]))
                all_score.append(np.array([]))

        # demo purpose
        if opt.demo:
            cv2.imwrite(os.path.join(out_dir, '{:05d}.jpg'.format(dataset.frame)), im0)
            vid_writer.write(im0)

    pbar.close()

    npy_bbox = np.array(all_bbox)
    npy_score = np.array(all_score)
    save_trackBbox = os.path.splitext(vid_path)[0] + '_obj_bbox.npy'
    save_trackScore = os.path.splitext(vid_path)[0] + '_obj_score.npy'
    np.save(save_trackBbox, npy_bbox)
    np.save(save_trackScore, npy_score)
    print('person bbox info save at:', save_trackScore)

if __name__ == '__main__':
    opt = get_opt()
    with torch.no_grad():
        obj_detect(opt)
