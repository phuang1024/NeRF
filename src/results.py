import cv2
import matplotlib.pyplot as plt
from tqdm import trange

from dataset import *
from model import *
from utils import *


def load_latest_model(args):
    model = NeRF(3).to(DEVICE)
    path = get_last_model(get_last_run(args))
    print("Loading model from", path)
    model.load_state_dict(torch.load(path))
    return model


def run_nerf(model, loc, rot, fov):
    image = render_image(model, loc, rot, fov, (64, 64))
    image = image.detach().cpu().numpy()
    image = np.clip(image*255, 0, 255).astype(np.uint8)
    return image


def compare(dataset, model):
    """
    Plot comparisons of render and ground truth.
    """
    num_samples = 8
    with torch.no_grad():
        for i in trange(num_samples):
            meta = dataset.images[i][1]
            loc = torch.tensor(meta["loc"], device=DEVICE, dtype=torch.float32)
            rot = torch.tensor(meta["rot"], dtype=torch.float32)

            image = run_nerf(model, loc, rot, meta["fov"])

            truth = cv2.imread(str(dataset.images[i][0]))
            truth = cv2.cvtColor(truth, cv2.COLOR_BGR2RGB)

            plt.subplot(2, num_samples, i+1)
            plt.imshow(image, aspect="auto")
            plt.axis("off")

            plt.subplot(2, num_samples, i+num_samples+1)
            plt.imshow(truth, aspect="auto")
            plt.axis("off")

    plt.savefig("compare.png")
    plt.show()


def turntable(model, steps=60, output="turntable.mp4"):
    """
    Create turntable animation.
    """
    video = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))

    for i in trange(steps):
        theta = i / steps * 2 * np.pi
        phi = np.pi / 2
        r = 3
        loc = np.array([
            r * np.cos(theta) * np.sin(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(phi)
        ])
        rot = ray_to_rot(-loc)

        loc = torch.tensor(loc, device=DEVICE, dtype=torch.float32)
        rot = torch.tensor(rot, dtype=torch.float32)

        image = run_nerf(model, loc, rot, math.radians(60))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video.write(image)

    video.release()


def main():
    parser = create_parser()
    parser.add_argument("action", choices=["compare", "turntable"])
    args = parser.parse_args()

    dataset = ImageDataset(args.data)
    model = load_latest_model(args)

    if args.action == "compare":
        compare(dataset, model)
    elif args.action == "turntable":
        turntable(model)


if __name__ == "__main__":
    main()
