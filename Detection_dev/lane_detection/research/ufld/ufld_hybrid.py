import torch, os, cv2, glob
import numpy as np
from utils.common import merge_config, get_model
import torchvision.transforms as transforms
from PIL import Image


def pred2coords(pred, row_anchor, col_anchor, local_width=1, original_image_width=1640, original_image_height=590):
    max_indices_row = pred['loc_row'].argmax(1).cpu()
    valid_row = pred['exist_row'].argmax(1).cpu()
    max_indices_col = pred['loc_col'].argmax(1).cpu()
    valid_col = pred['exist_col'].argmax(1).cpu()

    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()

    coords = []
    row_lane_idx = [1, 2]
    col_lane_idx = [0, 3]

    for i in row_lane_idx:
        tmp = []
        if valid_row[0, :, i].sum() > pred['loc_row'].shape[2] / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0, k, i]:
                    all_ind = torch.tensor(list(range(max(0, max_indices_row[0, k, i] - local_width),
                                                      min(pred['loc_row'].shape[1] - 1,
                                                          max_indices_row[0, k, i] + local_width) + 1)))
                    out_tmp = (pred['loc_row'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (pred['loc_row'].shape[1] - 1) * original_image_width
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
            coords.append(tmp)

    for i in col_lane_idx:
        tmp = []
        if valid_col[0, :, i].sum() > pred['loc_col'].shape[2] / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0, k, i]:
                    all_ind = torch.tensor(list(range(max(0, max_indices_col[0, k, i] - local_width),
                                                      min(pred['loc_col'].shape[1] - 1,
                                                          max_indices_col[0, k, i] + local_width) + 1)))
                    out_tmp = (pred['loc_col'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (pred['loc_col'].shape[1] - 1) * original_image_height
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            coords.append(tmp)
    return coords


def main():
    DIR_IN = "data_input"
    DIR_OUT = "results/03_ufld_hybrid"

    if not os.path.exists(DIR_OUT):
        os.makedirs(DIR_OUT)

    torch.backends.cudnn.benchmark = True
    args, cfg = merge_config()
    cfg.batch_size = 1

    net = get_model(cfg)
    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    dict_fix = {k[7:] if 'module.' in k else k: v for k, v in state_dict.items()}
    net.load_state_dict(dict_fix, strict=False)
    net.cuda().eval()

    img_transforms = transforms.Compose([
        transforms.Resize((cfg.train_height, cfg.train_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    files = glob.glob(os.path.join(DIR_IN, "*.png")) + glob.glob(os.path.join(DIR_IN, "*.jpg"))
    print(f"Przetwarzanie hybrydowe z {DIR_IN}...")

    for f_path in files:
        img_name = os.path.basename(f_path)
        frame = cv2.imread(f_path)
        h, w = frame.shape[:2]

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = img_transforms(img_pil).unsqueeze(0).cuda()

        with torch.no_grad():
            pred = net(img_tensor)

        coords = pred2coords(pred, cfg.row_anchor, cfg.col_anchor, w, h)

        road_mask = np.array([
            [int(w * 0.10), h], [int(w * 0.65), h],
            [int(w * 0.55), int(h * 0.40)], [int(w * 0.40), int(h * 0.40)]
        ], np.int32)

        for lane in coords:
            valid_pts = [p for p in lane if cv2.pointPolygonTest(road_mask, (float(p[0]), float(p[1])), False) >= 0]

            if len(valid_pts) > 5:
                pts = np.array(valid_pts)
                slope, intercept = np.polyfit(pts[:, 1], pts[:, 0], 1)

                if abs(slope) < 1.2:
                    y1, y2 = h, int(h * 0.42)
                    x1, x2 = int(slope * y1 + intercept), int(slope * y2 + intercept)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 6)

        cv2.imwrite(os.path.join(DIR_OUT, img_name), frame)
        print(f"Zapisano: {img_name}")


if __name__ == "__main__":
    main()