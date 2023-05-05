import torch
import torch.cuda.nvtx as nvtx
import argparse
from pynvml.smi import nvidia_smi

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("-O", "--num-operations", help="the total number of MAC operations", type=int, default=1024 * 1024 * 1024 * 16)
parser.add_argument("-N", "--num-iterations", help="the number of test iterations", type=int, default=1024)
parser.add_argument("--power-samples", help="the number of power samples per experiment", type=int, default=100)
args = parser.parse_args()
print(args)


def rand(dtype, n):
    """
    Generates a random tensor of args.batch_size nxn matrices of the given data type
    :param dtype:
    :param n:
    :return:
    """

    B = args.num_operations // (n * n * n)

    if dtype == torch.int16:
        return torch.randint(low=-(1 << 15), high=(1 << 15), size=(B, n, n), device='cuda', dtype=dtype)
    elif dtype == torch.int32:
        return torch.randint(low=-(1 << 31), high=(1 << 31), size=(B, n, n), device='cuda', dtype=dtype)
    else:
        return torch.rand(size=(B, n, n), device='cuda', dtype=dtype)


def testMatrixMult(n, dtype, output):
    """
    Tests the batched matrix multiplication (B x n x n) operation on data of the given data type
    :param n:
    :param dtype:
    :param output:
    :return:
    """

    x = rand(dtype, n)
    y = rand(dtype, n)

    torch.cuda.synchronize()

    nvtx.range_push(f'Testing n={n} with {dtype}')
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    totalPower = 0
    numMeasurements = 0

    for i in range(args.num_iterations):
        torch.bmm(x, y)

        if i % (args.num_iterations // args.power_samples) == 0:
            totalPower += nvidia_smi.getInstance().DeviceQuery('power.draw')['gpu'][0]['power_readings']['power_draw']
            numMeasurements += 1
            torch.cuda.synchronize()

    end.record()
    torch.cuda.synchronize()

    nvtx.range_pop()

    output.write(f'Tested n={n} with {dtype}: {round(totalPower / numMeasurements, 0)}W and {round(start.elapsed_time(end) / 1000, 2)}s\n')


if __name__ == '__main__':

    output = open("output", "w")

    for n in [32, 128, 512, 2048]:
        testMatrixMult(n, torch.float32, output)

    output.close()
