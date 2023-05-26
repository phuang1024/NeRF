from dataset import *
from model import *
from utils import *


def load_latest_model(args):
    model = NeRF(3).to(DEVICE)
    path = get_last_model(get_last_run(args))
    model.load_state_dict(torch.load(path))
    return model


def main():
    args = create_parser().parse_args()

    dataset = ImageDataset(args.data)
    model = load_latest_model(args)

    meta, _, _ = dataset.get_meta(0)
    loc = torch.tensor(meta["loc"], device=DEVICE, dtype=torch.float32)
    rot = torch.tensor(meta["rot"], dtype=torch.float32)
    image = render_image(model, loc, rot, 60, (128, 128))
    image = image.detach().cpu().numpy()
    image = np.clip(image*255, 0, 255).astype(np.uint8)
    cv2.imwrite("image.png", image)


if __name__ == "__main__":
    main()
