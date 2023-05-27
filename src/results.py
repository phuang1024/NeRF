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

    with torch.no_grad():
        meta, _, _ = dataset.get_meta(0)
        loc = torch.tensor(meta["loc"], device=DEVICE, dtype=torch.float32)
        rot = torch.tensor(meta["rot"], dtype=torch.float32)
        image = render_image(model, loc, rot, math.radians(60), (64, 64))
        image = image.detach().cpu().numpy()
        image = np.clip(image*255, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite("image.png", image)


if __name__ == "__main__":
    main()
