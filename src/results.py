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


def main():
    args = create_parser().parse_args()

    dataset = ImageDataset(args.data)
    model = load_latest_model(args)

    num_samples = 8
    with torch.no_grad():
        for i in trange(num_samples):
            meta = dataset.images[i][1]
            loc = torch.tensor(meta["loc"], device=DEVICE, dtype=torch.float32)
            rot = torch.tensor(meta["rot"], dtype=torch.float32)

            image = render_image(model, loc, rot, meta["fov"], (64, 64))
            image = image.detach().cpu().numpy()
            image = np.clip(image*255, 0, 255).astype(np.uint8)

            truth = cv2.imread(str(dataset.images[i][0]))
            truth = cv2.cvtColor(truth, cv2.COLOR_BGR2RGB)

            plt.subplot(2, num_samples, i+1)
            plt.imshow(image, aspect="auto")
            plt.axis("off")

            plt.subplot(2, num_samples, i+num_samples+1)
            plt.imshow(truth, aspect="auto")
            plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()
