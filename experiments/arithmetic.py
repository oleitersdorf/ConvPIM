import torch
import torch.cuda.nvtx as nvtx
import argparse
from pynvml.smi import nvidia_smi

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num-dimensions", help="the number of dimensions for the test vectors", type=int, default=1024 * 1024 * 1024)
parser.add_argument("-N", "--num-iterations", help="the number of test iterations", type=int, default=1024)
parser.add_argument("--power-samples", help="the number of power samples per experiment", type=int, default=100)
args = parser.parse_args()
print(args)


def rand(dtype):
    """
    Generates a random vector of length args.num_dimensions of the given data type
    :param dtype:
    :return:
    """

    if dtype == torch.int16:
        return torch.randint(low=-(1 << 15), high=(1 << 15), size=(args.num_dimensions, ), device='cuda', dtype=dtype)
    elif dtype == torch.int32:
        return torch.randint(low=-(1 << 31), high=(1 << 31), size=(args.num_dimensions, ), device='cuda', dtype=dtype)
    else:
        return torch.rand(size=(args.num_dimensions, ), device='cuda', dtype=dtype)


def testVectorOp(operation, dtype, output):
    """
    Tests the vectored operation on data of the given data type
    :param operation:
    :param dtype:
    :param output:
    :return:
    """

    x = rand(dtype)
    y = rand(dtype)

    torch.cuda.synchronize()

    nvtx.range_push(f'Testing {operation.__name__} with {dtype}')
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    totalPower = 0
    numMeasurements = 0

    for i in range(args.num_iterations):
        operation(x, y)

        if i % (args.num_iterations // args.power_samples) == 0:
            totalPower += nvidia_smi.getInstance().DeviceQuery('power.draw')['gpu'][0]['power_readings']['power_draw']
            numMeasurements += 1
            torch.cuda.synchronize()

    end.record()
    torch.cuda.synchronize()

    nvtx.range_pop()

    output.write(f'Tested {operation.__name__} with {dtype}: {round(totalPower / numMeasurements, 0)}W and {round(start.elapsed_time(end) / 1000, 2)}s\n')


if __name__ == '__main__':

    output = open("output", "w")

    for op in [torch.add, torch.mul]:
        for dtype in [torch.int16, torch.int32, torch.float16, torch.float32]:
            testVectorOp(op, dtype, output)

    output.close()
