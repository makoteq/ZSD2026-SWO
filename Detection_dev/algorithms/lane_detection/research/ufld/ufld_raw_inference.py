import torch, os, cv2, glob
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
    DIR_OUT = "results/01_ufld_raw"

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

    print(f"Przetwarzanie plików z {DIR_IN}...")

    for f_path in files:
        img_name = os.path.basename(f_path)
        img_pil = Image.open(f_path).convert('RGB')
        w, h = img_pil.size

        img_tensor = img_transforms(img_pil).unsqueeze(0).cuda()

        with torch.no_grad():
            pred = net(img_tensor)

        coords = pred2coords(pred, cfg.row_anchor, cfg.col_anchor, original_image_width=w, original_image_height=h)

        vis = cv2.imread(f_path)
        for lane in coords:
            for pt in lane:
                if 0 <= pt[0] < w and 0 <= pt[1] < h:
                    cv2.circle(vis, pt, 4, (0, 255, 0), -1)

        cv2.imwrite(os.path.join(DIR_OUT, img_name), vis)
        print(f"Zapisano: {img_name}")


if __name__ == "__main__":
    main()