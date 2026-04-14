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
    DIR_OUT = "results/02_ufld_ipm"

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
    print(f"Przetwarzanie IPM z {DIR_IN}...")

    for f_path in files:
        img_name = os.path.basename(f_path)
        frame = cv2.imread(f_path)
        h, w = frame.shape[:2]

        src = np.float32([
            [int(w * 0.15), h],
            [int(w * 0.60), h],
            [int(w * 0.55), int(h * 0.35)],
            [int(w * 0.45), int(h * 0.35)]
        ])

        dst = np.float32([
            [int(w * 0.10), h],
            [int(w * 0.90), h],
            [int(w * 0.65), int(h * 0.25)],
            [int(w * 0.35), int(h * 0.25)]
        ])

        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(frame, matrix, (w, h), flags=cv2.INTER_LANCZOS4)

        img_pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        img_tensor = img_transforms(img_pil).unsqueeze(0).cuda()

        with torch.no_grad():
            pred = net(img_tensor)

        coords = pred2coords(pred, cfg.row_anchor, cfg.col_anchor, original_image_width=w, original_image_height=h)

        for lane in coords:
            pts = np.array([pt for pt in lane if 0 <= pt[0] < w and 0 <= pt[1] < h], np.int32)
            if len(pts) > 1:
                cv2.polylines(warped, [pts.reshape((-1, 1, 2))], False, (0, 255, 0), 5)

        cv2.imwrite(os.path.join(DIR_OUT, img_name), warped)
        print(f"Zapisano: {img_name}")

if __name__ == "__main__":
    main()