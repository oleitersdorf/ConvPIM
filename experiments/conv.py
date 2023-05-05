import torch
import torch.cuda.nvtx as nvtx
import argparse
from pynvml.smi import nvidia_smi
from tqdm import tqdm

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("-B", "--batch-size", help="the number of images in a batch", type=int, default=1024 * 32)
parser.add_argument("-W", "--image-width", help="the width of the image", type=int, default=224)
parser.add_argument("-H", "--image-height", help="the height of the image", type=int, default=224)
parser.add_argument("-N", "--num-iterations", help="the number of test iterations", type=int, default=128)
parser.add_argument("--power-samples", help="the number of power samples per experiment", type=int, default=100)
args = parser.parse_args()
print(args)


def rand(dtype, size):
    """
    Generates a random tensor of the given size and type
    :param dtype:
    :param size:
    :return:
    """

    if dtype == torch.int16:
        return torch.randint(low=-(1 << 15), high=(1 << 15), size=size, device='cuda', dtype=dtype)
    elif dtype == torch.int32:
        return torch.randint(low=-(1 << 31), high=(1 << 31), size=size, device='cuda', dtype=dtype)
    else:
        return torch.rand(size=size, device='cuda', dtype=dtype)


def testConv(k, dtype, output):
    """
    Tests the batched convolution operation on data of the given data type with a kernel of size kxk
    :param k:
    :param dtype:
    :param output:
    :return:
    """

    A = rand(dtype, (args.batch_size, 1, args.image_width, args.image_height))
    K = rand(dtype, (1, 1, k, k))

    torch.cuda.synchronize()

    nvtx.range_push(f'Testing k={k} with {dtype}')
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    totalPower = 0
    numMeasurements = 0

    for i in tqdm(range(args.num_iterations)):
        torch.nn.functional.conv2d(A, K)

        if i % (args.num_iterations // args.power_samples) == 0:
            totalPower += nvidia_smi.getInstance().DeviceQuery('power.draw')['gpu'][0]['power_readings']['power_draw']
            numMeasurements += 1
            torch.cuda.synchronize()

    end.record()
    torch.cuda.synchronize()

    nvtx.range_pop()

    output.write(f'Tested k={k} with {dtype}: {round(totalPower / numMeasurements, 0)}W and {round(start.elapsed_time(end) / 1000, 2)}s\n')


if __name__ == '__main__':

    output = open("output", "w")
    
    torch.backends.cudnn.allow_tf32 = False

    for k in [1, 3, 5, 7, 9]:
        testConv(k, torch.float32, output)

    output.close()
